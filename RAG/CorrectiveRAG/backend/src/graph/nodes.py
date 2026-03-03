import os, re, logging
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain.schema import Document

from backend.src.graph.states import CRAGState, Is_Retrieval, client_chat_oai_with_structured_output_retrieval, \
     client_chat_oai_with_structured_output_relevance, client_chat_oai_with_structured_output_keep_sentence, client_chat_oai

from langchain.schema import SystemMessage, HumanMessage
from langchain_community.tools import TavilySearchResults

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

token_provider = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("TOKEN_PROVIDER"))
# Set Tavily API key
os.environ["TAVILY_API_KEY"] = "..."

# Node to check if document retrieval is needed based on the query or not
def is_retrieve_node(state:CRAGState):
    query = state["query"]
    logger.info("[is_retrieve_node] Evaluating retrieval need. query=%s", query)
    prompt = [
        SystemMessage(content="""You decide whether external retrieval is required.
Return True only when answering requires specific, up-to-date, niche, or source-grounded information.
Return False when the query can be answered reliably with general knowledge, definitions, or common concepts.
Be conservative: if unsure, return True."""),
        HumanMessage(content=f"""Query: {query}""")
    ]
                     
    response = client_chat_oai_with_structured_output_retrieval.invoke(prompt)
    logger.info("[is_retrieve_node] is_retrieval_needed=%s", response.is_retrieval)
    return {"is_retrieval_needed": response.is_retrieval}

# Router to decide where to go after checking if retrieval is needed or not
def router_after_is_retrieve_node(state:CRAGState):
    if state["is_retrieval_needed"]:
        return "retrieve"
    else:
        return "not_retrieve"

# Node to retrieve top 3 similar documents
def retrieve_node(state:CRAGState):
    query = state["query"]
    logger.info("[retrieve_node] Retrieving docs. query=%s", query)
    embeddings = AzureOpenAIEmbeddings(
            azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"), 
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_ad_token_provider = token_provider
                                        ) 
        
    # Initialize Azure Search vector store
    vector_store = AzureSearch(azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"), 
                                azure_search_key = os.getenv("AZURE_SEARCH_API_KEY"), 
                                index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                                embedding_function = embeddings.embed_query
                                    )
    retrieved_docs = vector_store.similarity_search(query, k=3)
    logger.info("[retrieve_node] Retrieved %d docs", len(retrieved_docs))
    return {"retrieved_docs": retrieved_docs}

# Tavily Search Tool instance
web_search_tool = TavilySearchResults(max_results=3)

# Web Search Function
def web_search(query: str) -> str:
    """Search the web for information about the given query."""
    logger.info("[web_search] Running Tavily search. query=%s", query)
    search_results = web_search_tool.invoke(query)
    
    if search_results:
        web_context = "\n\n".join([
            f"Source: {result.get('url', 'N/A')}\nContent: {result.get('content', '')}" 
            for result in search_results
        ])
    else:
        web_context = "No web results found."
    logger.info("[web_search] Results received=%d", len(search_results) if search_results else 0)
    
    return web_context



# Node to re qrite query to make it more specific and clear for web search
def rewrite_query(state:CRAGState):
    query = state["query"]
    logger.info("[rewrite_query] Rewriting query for web search. original_query=%s", query)
    prompt = [
        SystemMessage(content="""Rewrite the query for web search.
Requirements:
1) Preserve original intent.
2) Clarify wording without changing meaning.
3) Remove ambiguity and pronouns.
4) Return a single concise search query string only.
5) Do NOT add new facts, dates, years, locations, entities, constraints, or assumptions that are not explicitly in the original query.
6) If the original query has no timeframe, do not introduce one.
Do not add explanations, bullets, or quotes."""),
        HumanMessage(content=f"""Original Query: {query}\n\n""")
    ]
    response = client_chat_oai.invoke(prompt)
    rewritten_query = (response.content or "").strip()

    # Safety check: prevent adding explicit years when original query has none
    original_has_year = bool(re.search(r'\b(19|20)\d{2}\b', query))
    if not original_has_year:
        rewritten_query = re.sub(r'\b(19|20)\d{2}\b', '', rewritten_query)
        rewritten_query = re.sub(r'\s{2,}', ' ', rewritten_query).strip(" ,.-")

    logger.info("[rewrite_query] rewritten_query=%s", rewritten_query)
    return {"web_query": rewritten_query}

# Node to perform web search and return web context
def web_search_node(state:CRAGState):
    # Use web query if available, otherwise use original query for web search
    web_query = state["web_query"] or state["query"]
    web_context = web_search(web_query)
    
    # Convert web results to Document objects
    web_docs = [Document(page_content=web_context, metadata={"source": "web_search"})]
    
    # For Ambiguous verdict: combine existing filtered_docs with web_docs
    # For Bad verdict: filtered_docs will be empty, so only web_docs are used
    existing_filtered_docs = state.get('relevant_docs', [])
    combined_docs = existing_filtered_docs + web_docs
    logger.info(
        "[web_search_node] web_query=%s existing_docs=%d combined_docs=%d web_context_chars=%d",
        web_query,
        len(existing_filtered_docs),
        len(combined_docs),
        len(web_context or "")
    )
    return {"relevant_docs": combined_docs, "web_context": web_context}

# Node to check if document is relevant or not to answer the query
def is_doc_relevant_node(state: CRAGState):
    UPPER_THRESHOLD = 0.7
    LOWER_THRESHOLD = 0.3

    query = state["query"]
    retrieved_docs = state["retrieved_docs"]
    relevant_docs = state["relevant_docs"]
    verdict = state["verdict"]
    logger.info("[is_doc_relevant_node] Scoring relevance for %d docs", len(retrieved_docs))

    all_scores = []
    all_reasons = []

    for doc in retrieved_docs:
        prompt = [
            SystemMessage(content="""Score relevance of the document to the query from 0 to 1.
Use this rubric:
- 0.8 to 1.0: directly answers core parts of the query with specific facts.
- 0.5 to 0.79: partially relevant; useful supporting context but incomplete.
- 0.0 to 0.49: mostly off-topic, too generic, or insufficient for answering.
Do not assume missing details (such as year, geography, or product variants) that are not in the query or document.
Return an honest score and a short reason grounded in document content."""),
            HumanMessage(content=f"""Query: {query}\n\nRetrieved Document: {doc.page_content}""")
        ]
        
        response = client_chat_oai_with_structured_output_relevance.invoke(prompt)
        all_scores.append(response.score)
        all_reasons.append(response.reason)

        if response.score >= UPPER_THRESHOLD:
            relevant_docs.append(doc)
    
    if any(score >= UPPER_THRESHOLD for score in all_scores):
        verdict = "Good"
    elif all(score <= LOWER_THRESHOLD for score in all_scores):
        verdict = "Bad"
    else:
        verdict = "Ambiguous"

    logger.info(
        "[is_doc_relevant_node] verdict=%s kept_docs=%d scores=%s",
        verdict,
        len(relevant_docs),
        [round(score, 3) for score in all_scores]
    )

    return {"relevant_docs": relevant_docs, "verdict": verdict, "doc_scores": all_scores, "reason": all_reasons}

def route_after_is_doc_relevant_node(state:CRAGState):
    logger.info("[route_after_is_doc_relevant_node] verdict=%s", state["verdict"])
    if state["verdict"] == "Good":
        return "refine_node"
    else :
        return "rewrite_query"
    
# Function to split docs into sentences
def sentence_breaker(text):
    # Split on sentence boundaries and newlines; keep non-empty segments
    sentences = re.split(r'(?<=[.!?])\s+|\n+', text)
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    return cleaned

# Node to check if sentence is relevant or not and if yes then add it to refined context else discard
def refine_node(state: CRAGState):
    query = state["query"]
    relevant_docs = state.get("relevant_docs", [])
    logger.info("[refine_node] Starting refinement. relevant_docs=%d", len(relevant_docs))
    relevant_sentences = []
    sentences = []
    for doc in relevant_docs:
        page_text = (doc.page_content or "").strip()
        if page_text:
            sentences.extend(sentence_breaker(page_text))

    logger.info("[refine_node] Candidate sentences=%d", len(sentences))

    for sent in sentences:

        prompt = [
            SystemMessage(content="""Decide if this sentence should be kept for final answer generation.
Keep the sentence only if it provides factual information that directly helps answer the query.
Reject sentences that are generic, repetitive, opinionated, promotional, or irrelevant.
Do not rely on assumptions not present in the query or sentence.
Prefer precision over recall."""),
            HumanMessage(content=f"""Query: {query}\n\nDocument Sentence: {sent}""")
        ]
        response = client_chat_oai_with_structured_output_keep_sentence.invoke(prompt)

        if response.keep:
            relevant_sentences.append(sent)

    if relevant_sentences:
        context = " ".join(relevant_sentences)
    else:
        # Fallback so context is not blank when relevant documents are present
        context = "\n\n".join(
            [(doc.page_content or "").strip() for doc in relevant_docs if (doc.page_content or "").strip()]
        )

    logger.info(
        "[refine_node] kept_sentences=%d refined_context_chars=%d",
        len(relevant_sentences),
        len(context or "")
    )

    return {"refined_context": context}



def generate_answer_node(state: CRAGState):
    query = state["query"]
    refined_context = state["refined_context"]
    logger.info("[generate_answer_node] Generating answer. refined_context_chars=%d", len(refined_context or ""))
    
    #final_context = "\n\n".join([refined_context])

    prompt = [
        SystemMessage(content="""Answer the query using only the provided refined_context.
Requirements:
1) Use only facts supported by refined_context.
2) If context is partial, provide the best possible answer from available evidence and clearly state uncertainty.
3) Say 'I don't know' only if refined_context has no usable facts for the query.
    4) Keep the answer concise and factual; do not invent details.
    5) Do not introduce unstated specifics such as dates/years, regions, or constraints unless explicitly present in refined_context or query."""),
        HumanMessage(content=f"""    Query: {query}\n\n
                                     refined_context: {refined_context}""")
    ]

    response = client_chat_oai.invoke(prompt)
    logger.info("[generate_answer_node] Answer generated. answer_chars=%d", len(response.content or ""))
    return {"answer": response.content}


  

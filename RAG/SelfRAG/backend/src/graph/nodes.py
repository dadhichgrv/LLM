
import os
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain.schema import SystemMessage, HumanMessage
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from backend.src.graph.state import SelfRAGState, client_chat_oai_shud_retrieve, client_chat_oai, client_chat_oai_relevance_score

from dotenv import load_dotenv
load_dotenv()

token_provider = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("TOKEN_PROVIDER"))

# This node decides if retrieval is needed based on the query. 
# It uses a structured output LLM to return a boolean value indicating whether retrieval is needed or not.
def shud_retrieve_node(state: SelfRAGState):
    query = state["query"]
    prompt = [
        SystemMessage(content="""You are a helpful assistant who decides if a query needs to
                      retrieve information from a knowledge base to answer the question. 
                      If the query can be answered without retrieval by using the parametric
                      knowledge, respond with "False". 
                      If retrieval is needed, respond with "True"."""),
        HumanMessage(content=f"""query: {query}""")
    ]

    response = client_chat_oai_shud_retrieve.invoke(prompt)
    return {"retrieve_or_not": response.keep}

# This is a router function to decide which node to route
def route_after_evaluation(state: SelfRAGState):
    if state["retrieve_or_not"]:
        return "retrieve_node"
    else:
        return "generate_answer_node"

# This node generates an answer to the query without retrieval, using only the parametric knowledge of the LLM.
def generate_answer_node(state: SelfRAGState):
    query = state["query"]
    prompt = [
        SystemMessage(content="""You are a helpful assistant who answers the question based on the 
                      question asked. If you are not sure, say that I don't know based on my knowledge. 
                      """),
        HumanMessage(content=f"query: {query}")
    ]

    response = client_chat_oai.invoke(prompt)
    return {"answer": response.content}

# This node retrieves relevant information from the Azure Search vector store based on the query, 
# and generates an answer using the retrieved information.

def retrieve_node(state: SelfRAGState):

    retrieve_counter = state["retrieve_counter"] + 1
    

    if retrieve_counter > 5:
        return {"answer": "Sorry, Maximum trials for retrieval reached.I am having trouble retrieving relevant information."}
    else:

        query = state["query"] 
        
        # Create embeddings client used by AzureSearch vector store.
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
        
        similar_docs = vector_store.similarity_search(query, k=3)

    return {"retrieved_docs": similar_docs, "retrieve_counter": state["retrieve_counter"]}


def check_docs_relevant_node(state: SelfRAGState):
    THRESHOLD = 0.5
    query = state["query"] 
    retrieved_docs = state["retrieved_docs"]
    final_docs = []
    final_scores = []
    final_reasons = []

    for doc in retrieved_docs:
        prompt = [
            SystemMessage(content=""" You are a helpful assistant who checks if the given document is relevant to answer the query.
                                      A document is relevant if it contains information that can be used to answer the query.
                                      Generate a relevance score between 0 and 1 and a reason for the score"""),
            HumanMessage(content=f"""query: {query}
                                     document: {doc.page_content}"""
                          )
                        ]
        response = client_chat_oai_relevance_score.invoke(prompt)

        final_scores.append(response.score)
        final_reasons.append(response.reason)

        if response.score >= THRESHOLD:
            final_docs.append(doc)
            state["relevant_or_not"] = True
        
        context = "\n".join([doc.page_content for doc in final_docs])
    
    return {"relevant_docs": final_docs, "context": context, "final_scores": final_scores, 
            "final_reasons": final_reasons, "relevant_or_not": state["relevant_or_not"]}

def router_after_checking_relevant_docs(state: SelfRAGState):
    if state["relevant_docs"] and len(state["relevant_docs"]) > 0:
        return "relevant"
    else:
        return "not_relevant"

def no_answer_found(state: SelfRAGState):
    return {"answer": "Sorry, I could not find relevant information to answer your question."}

# Thos node generates answer to the query based on the retrieved relevant documents. 
# It also has a counter to limit the number of times it can be invoked to prevent infinite loops 
# in case the generated answer is not grounded on the retrieved information.

def generate_answer_with_retrieved_docs_node(state: SelfRAGState):
    
    generate_answer_counter = state["generate_answer_counter"] + 1
    if generate_answer_counter > 5:
        return {"answer": "Sorry, Maximum trials for generating answer reached.I am having trouble generating an answer based on the retrieved information."}
    else:

        query = state["query"]
        context = state["context"]
        prompt = [
            SystemMessage(content="""You are a helpful assistant who answers the question based on the 
                        question asked and the retrieved information.  
                        """),
            HumanMessage(content=f"""query: {query}
                                    context: {context}""")
        ]


        response = client_chat_oai.invoke(prompt)
    return {"answer": response.content, "generate_answer_counter": generate_answer_counter}


def is_grounded_node(state:SelfRAGState):
    answer = state["answer"]
    context = state["context"]
    query = state["query"]

    prompt = [
        SystemMessage(content="""You are a helpful assistant who checks if the answer is grounded on the retrieved information.
                            Check if the information in answer can be found in the context. 
                           If it can be found, respond with "True". If answer is mostly unrelated to the query or unsupported, 
                          respond with "False".
                      How to decide is it is grounded : Every meaningful claim should be suported by context and answer does not introduce
                      any qualitative / interceptive words that are not present in the context.
                      Do not use outside knowledge that is not present in the context to determine if the answer is grounded.
                      If it is partially supported by the context, respond with "False" as well since the answer is not fully grounded.
                      """),
        HumanMessage(content=f"""answer: {answer},
                                 query: {query},
                                 context: {context}""")                   
    ]

    response = client_chat_oai_shud_retrieve.invoke(prompt)

    return {"is_response_grounded": response.keep}


def router_after_grounded_check(state: SelfRAGState):
    if state["is_response_grounded"]:
        return "grounded"
    else:
        return "not_grounded"


def revise_answer_node(state: SelfRAGState):
    
    revise_counter = state["revise_counter"] + 1
    if revise_counter > 5:
        return {"answer": "Sorry, Maximum trials for revising answer reached.I am having trouble revising the answer to make it more grounded."}
    else:

        query = state["query"]
        context = state["context"]
        answer = state["answer"]

        prompt = [
            SystemMessage(content="""You are a helpful assistant who revises the answer to make it more grounded on the retrieved information.
                                Revise the answer to make it more grounded on the retrieved information in the context."""),
            HumanMessage(content=f"""query: {query}
                                    context: {context}
                                    existing answer: {answer}""")
        ]

        response = client_chat_oai.invoke(prompt)
    return {"answer": response.content, "revise_counter": revise_counter}    
    
def is_useful_node(state: SelfRAGState):
    query = state["query"]
    answer = state["answer"]

    prompt = [
        SystemMessage(content="""You are a helpful assistant who judges if the answer is useful for the user query.
                            Check if the answer actually addresses the user query. 
                           If answer directly answers the query then it is useful, respond with "True". 
                          If answer is generic, off-topic , respond with "False"."""),
        HumanMessage(content=f"""query: {query}
                                 answer: {answer}""")
                ]
    response = client_chat_oai_shud_retrieve.invoke(prompt)
    return {"is_useful": response.keep}

def router_after_usefulness_check(state: SelfRAGState):
    if state["is_useful"]:
        return "useful"
    else:
        return "not_useful"
    
def rewrite_question_node(state: SelfRAGState):
    query = state["query"]

    prompt = [
        SystemMessage(content="""You are a helpful assistant who rewrites the user query to make it easier to answer based on the retrieved information.
                            Rewrite the question to make it easier to answer based on the retrieved information."""),
        HumanMessage(content=f"""Existing query: {query} 
                                Answer: {state["answer"]}""")]  

    response = client_chat_oai.invoke(prompt)
    original_query = state["query"]
    return {"query": response.content, "original_query": original_query}  


import requests, bs4, os, logging
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
# langchain-community==0.0.19 for WebBaseLoader error of pwd installation

# import langchain_experimental
# from langchain_experimental.text_splitter import SemanticChunker
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever # BM25 is best match re ranking algorithm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor, LLMChainFilter 
# LLM Chain Extractor will iterate over each document returned by the retriever and only extract content that is relevant to the query. 
# It will drop irrelevant documents 
# LLM Chain Filter will return only relevant documents without manipulating the content

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"....servicename.....")

# Load PDF and extract text content
url = "https://en.wikipedia.org/wiki/India"
loader = WebBaseLoader(web_paths=(url,),
                       bs_kwargs=dict(parse_only=bs4.SoupStrainer())
                       )
documents = loader.load()

print("File Loaded \n")

# Embeddings
azure_embeddings = AzureOpenAIEmbeddings(azure_deployment="....", 
                                                          azure_endpoint = "....endpoint name.....",
                                                          azure_ad_token_provider = token_provider)

# Use Semantic Chunking to create chunked documents
#semantic_chunker = SemanticChunker(embeddings = azure_embeddings)

# Create documents for responsibilities and qualifications document
#docs = semantic_chunker.create_documents([d.page_content for d in documents])

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
docs = text_splitter.split_documents(documents)

print("No of pages are ",len(docs))

faiss_index_pdf = FAISS.from_documents(docs, azure_embeddings)

retreiver = faiss_index_pdf.as_retriever(search_type="similarity",
                                          search_kwargs={"k":2})

query = "What is the definition of international poverty line"
print("\n Retrieved answer \n",retreiver.invoke(query))

llm = AzureChatOpenAI(model="...model...",
                      api_version = "...version...",
                      azure_endpoint = "...endpoint name....",
                      azure_ad_token_provider = token_provider)

mq_retriever = MultiQueryRetriever.from_llm(
    retriever=retreiver, 
    llm=llm,
    include_original = True
)

print("\n Multi Query generated queries \n")
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

print("\n Multi Query Retrieval \n", mq_retriever.invoke(query))

bm25_retriever = BM25Retriever.from_documents(docs)
print("\n BM25 Retriever \n",bm25_retriever.invoke(query))

print("\n Hybrid Search (BM25 + Semantic) \n")
ensemble_retriever = EnsembleRetriever( retrievers=[bm25_retriever,retreiver], weights = [0.7,0.3])
print(ensemble_retriever.invoke(query))

print("\n LLM Chain Extractor \n")
# Extract from each document only content that is relevant to the query
compressor = LLMChainExtractor.from_llm(llm=llm)
# retrieve documents similar to query and then applies to compressor
compressor_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                      base_retriever=retreiver)
print("\n Compression Retriever -> It will give relevant context from relevant document \n",compressor_retriever.invoke(query))

print("\n LLM Chain Filter \n")
_filter = LLMChainFilter.from_llm(llm=llm)
compression_retriever_filter = ContextualCompressionRetriever(base_compressor=_filter,
                                                              base_retriever=retreiver)
print("\n Compression Filter Retrieval --> It will give entire relevant document \n", compression_retriever_filter.invoke(query))



# # To save faiss embeddings
# faiss_index_pdf.save_local("faiss_index_pdf")

# # To load it again
# faiss_index_pdf = FAISS.load_local("faiss_index_pdf", 
#                                    embeddings = azure_embedings,
#                                    allow_dangerous_deserialization = True)


# # prompt = ChatPromptTemplate.from_template(""" Answer the following question based only on the the provided context.
# #                                           Think step by step before providing a detailed answer.
# #                                           <context>
# #                                           {context}
# #                                           </context>
# #                                            Question : {input}
# #                                           """)

# # from langchain.chains.combine_documents import create_stuff_documents_chain


# # # Document Chain has llm and prompt
# # # Document chain will take all the documents and format them into prompt and pass it to the LLM
# # document_chain = create_stuff_documents_chain(llm,prompt)


# # # crete retriever chain by combinig document chain and retreiver
# # from langchain.chains import create_retrieval_chain
# # retreival_chain = create_retrieval_chain(retreiver, document_chain)

# # def summary_bot(query):
# #   response = retreival_chain.invoke({"input":query})
# #   return response['answer']

# # query = "Give the gpa of the candidate "
# # print(summary_bot(query))

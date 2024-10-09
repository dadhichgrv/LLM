import logging
import azure
import os
import openai
import numpy as np
import pandas as pd
import azure.functions as func
from azure.keyvault.secrets import SecretClient
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient

from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader, TextLoader
import PyPDF2
from PyPDF2 import PdfReader
from langchain.prompts import ChatPromptTemplate

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"...service...")


# Load PDF and extract text content
loader = PyPDFLoader("Resume.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


# faiss_index_pdf = FAISS.from_documents(docs, AzureOpenAIEmbeddings(azure_deployment="...name...", 
#                                                               azure_endpoint = "...endpoint...",
#                                                                azure_ad_token_provider = token_provider))

# # To save faiss embeddings
# faiss_index_pdf.save_local("faiss_index_pdf")

# To load it again
faiss_index_pdf = FAISS.load_local("faiss_index_pdf", 
                                   embeddings = AzureOpenAIEmbeddings(azure_deployment="...name...", 
#                                                               azure_endpoint = "...endpoint...",
#                                                                azure_ad_token_provider = token_provider),
                                   allow_dangerous_deserialization = True)

query = "Give the gpa of the candidate "
query_answer = faiss_index_pdf.similarity_search(query)
#print(query_answer[0].page_content)

prompt = ChatPromptTemplate.from_template(""" Answer the following question based only on the the provided context.
                                          Think step by step before providing a detailed answer.
                                          <context>
                                          {context}
                                          </context>
                                           Question : {input}
                                          """)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(model="...name...",
                      api_version = "...version...",
                      azure_endpoint = "...endpoint...",
                      azure_ad_token_provider = token_provider)

# Document Chain has llm and prompt
# Document chain will take all the documents and format them into prompt and pass it to the LLM
document_chain = create_stuff_documents_chain(llm,prompt)

retreiver = faiss_index_pdf.as_retriever()


# crete retriever chain by combinig document chain and retreiver
from langchain.chains import create_retrieval_chain
retreival_chain = create_retrieval_chain(retreiver, document_chain)

def summary_bot(query):
  response = retreival_chain.invoke({"input":query})
  return response['answer']

query = "Give the gpa of the candidate "
print(summary_bot(query))


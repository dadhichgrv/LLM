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

# Open AI Key
# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"...service...")

# Load Text file and extract text content
pdf_file_path = 'cricket.txt'
loader = TextLoader(pdf_file_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=0)
docs = text_splitter.split_documents(documents)


faiss_index = FAISS.from_documents(docs, AzureOpenAIEmbeddings(azure_deployment="...name...", 
                                                               azure_endpoint = "...endpoint...",
                                                               azure_ad_token_provider = token_provider))
faiss_index.save_local("faiss_index")

query = "What is cricket"
query_answer = faiss_index.similarity_search(query)
print(query_answer[0].page_content)

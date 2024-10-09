# libraries
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from azure.identity import DefaultAzureCredential
from fastapi import FastAPI
from langserve import add_routes
import uvicorn
import requests
import streamlit as st

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"...service...")

llm = AzureChatOpenAI(
  model="...model...",
  api_version = "...version...",
  azure_endpoint = "...endpoint...",
  azure_ad_token_provider = token_provider
  )

# Prompt Template
prompt = ChatPromptTemplate.from_messages (
  [
    ("system","You are assistant to respond to the user queries"),
    ("user","Question:{question}")
  ]
)



parser = StrOutputParser()

# Create chain of prompt , llm and parser
chain = prompt|llm|parser


st.title("Question Answer")
input_text = st.text_input("Ask me anything")

if input_text:
    st.write(chain.invoke(input_text))


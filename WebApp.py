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

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"..service...")

llm = AzureChatOpenAI(
  model="...name...",
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

#prompt = ChatPromptTemplate.from_template("You are an assistant and you have to answer to the queries {question}")

# parser = StrOutputParser()

# # Create chain of prompt , llm and parser
# chain = prompt|llm|parser
# # Without parser you will get lot of other content in addition to the answer like id, model_name ... 

# answer = chain.invoke({'question':"What is RAG in AI"})
# print("Prompt ",prompt)
# print(answer)

# Create API
app = FastAPI(
    title = "Langchain Server",
    version = "1.0",
    description = "API Server"
    )

add_routes(app, llm, path="/openai" )

add_routes(app,prompt|llm,path="/query")

if __name__=="__main__":
    uvicorn.run(app, host="localhost", port = 8000)

# Just hit run arrow on top right to start langserve    



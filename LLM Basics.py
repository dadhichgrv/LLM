# libraries
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
import openai
from openai import AzureOpenAI
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from azure.identity import DefaultAzureCredential

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"...service...")

llm = AzureChatOpenAI(
  model="..model...",
  api_version = "....version....",
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

parser = StrOutputParser()

# Create chain of prompt , llm and parser
chain = prompt|llm|parser
# Without parser you will get lot of other content in addition to the answer like id, model_name ... 

answer = chain.invoke({'question':"The capital of India is"})
print("Prompt ",prompt)
print(answer)



# def chatbot(prompt):

#   response = llm.chat.completions.create(
#     model="..model..",
#     messages=prompt,
#     temperature=0.1,
#     max_tokens=500,
#     top_p=1,
#     frequency_penalty=0,
#     presence_penalty=0,
#     stop="\n"
#     )
  
#   return response.choices[0].message.content.strip()

# #p = [{"role": "user", "content": 'What is Retrieval Augmented Generation'}]

# #print(chatbot(p)) 



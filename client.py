# Creating front end that will interact with API that we have created on Langserve (WebApp.py file)

import requests
import streamlit as st

# Create function
def get_openai_response(input_text):
    response = requests.post("http://localhost:8000/query/invoke",
                             json={'input':{'question':input_text}}
                             )
    
    return response.json()['output']['content']


st.title("Question Answer")
input_text = st.text_input("Ask me anything")

if input_text:
    st.write(get_openai_response(input_text))

# Open new cmd terminal
# Go to cd "RAG Scripts"
# streamlit run client.py to start front end and type question    

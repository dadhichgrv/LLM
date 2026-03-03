import os
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from typing import Dict, List, Optional, Any, Literal, Annotated, TypedDict
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

token_provider = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("TOKEN_PROVIDER"))

class KeepSentence(BaseModel):
    keep : bool 

class RelevanceScore(BaseModel):
    score: float = Field(..., ge=0, le=1)
    reason : str

class Is_Retrieval(BaseModel):
    is_retrieval: bool

class CRAGState(TypedDict):
    query: str
    answer: str
    is_retrieval_needed: bool
    retrieved_docs: List[Document]
    web_context: str
    web_query: str
    refined_context: str
    relevant_docs: List[Document]
    verdict: Optional[Literal["Good", "Bad", "Ambiguous"]] 
    doc_scores : List[float]
    reason : List[str]
    


client_chat_oai = AzureChatOpenAI(   
                    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
                    azure_ad_token_provider=token_provider
                                        )   

client_chat_oai_with_structured_output_retrieval = client_chat_oai.with_structured_output(Is_Retrieval)
client_chat_oai_with_structured_output_relevance = client_chat_oai.with_structured_output(RelevanceScore)
client_chat_oai_with_structured_output_keep_sentence = client_chat_oai.with_structured_output(KeepSentence)
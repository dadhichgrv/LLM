import os
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_openai import AzureChatOpenAI
from typing import Dict, List, Optional, Any, Literal, Annotated, TypedDict
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

from dotenv import load_dotenv
load_dotenv()

token_provider = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("TOKEN_PROVIDER"))

class ShudRetrieveState(BaseModel):
    keep: bool

class RelevanceScore(BaseModel):
    score: float = Field(..., ge=0.0, le=1.0)
    reason: str
                           

class SelfRAGState(TypedDict):
    query: str
    original_query: str
    answer: str
    retrieve_or_not: bool
    retrieved_docs : List[Document]
    relevant_or_not : bool
    final_scores : List[float]
    final_reasons: List[str]
    relevant_docs : List[Document]
    context: str
    is_response_grounded: bool
    is_useful: bool
    retrieve_counter: int
    generate_answer_counter: int
    revise_counter: int


client_chat_oai = AzureChatOpenAI(   
                    model=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
                    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"), 
                    azure_ad_token_provider=token_provider
                                        )   

client_chat_oai_shud_retrieve = client_chat_oai.with_structured_output(ShudRetrieveState)
client_chat_oai_relevance_score = client_chat_oai.with_structured_output(RelevanceScore)


from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"....service....")

llm = AzureChatOpenAI(model="...model...",
                      api_version = "...version...",
                      azure_endpoint = "...endpoint....",
                      azure_ad_token_provider = token_provider)

# You can output the response in specific output format
class GameDetails(BaseModel):
    win_team : str = Field(description="The winning team in football game")
    lose_team : str = Field(description="The losing team in football game")
    venue : str = Field(description="The venue of football game")
    date : str = Field(description="Date on which the football game was played")
    score : str = Field(description="The score of football game")

parser = JsonOutputParser(pydantic_object=GameDetails)  

print(parser.get_format_instructions())

prompt_txt = """Who won the Champions league in 2021
                Use the following format when generating the output response
                
                Output format instructions:
                {format_instructions}
                """

prompt = PromptTemplate.from_template(template=prompt_txt)

llm_chain = (prompt | llm | parser)

response = llm_chain.invoke({"format_instructions":parser.get_format_instructions()})

print(response)

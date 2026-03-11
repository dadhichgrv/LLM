import base64 , os, logging
from pprint import pprint
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_community.tools import TavilySearchResults, tool
from langgraph.checkpoint.memory import MemorySaver

# Set token provider
token_provider = get_bearer_token_provider(
  DefaultAzureCredential(),"...")

# Set Tavily API key
os.environ["TAVILY_API_KEY"] = "..."

logger = logging.getLogger(__name__)


client_chat_oai = AzureChatOpenAI(   
   model="...",
   api_version="...",
   azure_endpoint="...", 
   azure_ad_token_provider=token_provider
)

# Tavily Search Tool instance
web_search_tool = TavilySearchResults(max_results=3)

image_location = "Food.jpg"

# Read image and convert to Base64
with open(image_location, "rb") as image_file:
    # Convert to bytes, then to Base64 string
    image_bytes = bytes(image_file.read())
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")


@tool
def web_search(query: str) -> str:
    """Search the web for information about the given query."""
    logger.info("[web_search] Running Tavily search. query=%s", query)
    search_results = web_search_tool.invoke(query)
    
    if search_results:
        web_context = "\n\n".join([
            f"Source: {result.get('url', 'N/A')}\nContent: {result.get('content', '')}" 
            for result in search_results
        ])
    else:
        web_context = "No web results found."
    logger.info("[web_search] Results received=%d", len(search_results) if search_results else 0)
    
    return web_context

tools = [web_search]
client_chat_oai_with_tools = client_chat_oai.bind_tools(tools)

prompt_image = [
    SystemMessage(content="""You are a MasterChef who can make tasty Indian food. 
                  You are given ingredients from which you have to make a dish.
                  You have web search tool to search for recipes 
                  Return recipe suggestions and recipe instructions.
                  """) ,
   
    HumanMessage(content=[
                 {"type": "text", "text": "What can I make from them."},
                 {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}}
                 ])
        ]

checkpointer = MemorySaver()
thread_id = "1"
config = {"configurable": {"thread_id": thread_id}}

response = client_chat_oai_with_tools.invoke(prompt_image, config=config)
pprint("Messages \n")
pprint(response)
pprint("Response content: \n")
pprint(response.content)

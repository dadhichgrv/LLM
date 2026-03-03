# Import Libraries
import os, logging, tempfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import AzureSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from azure.storage.blob import BlobServiceClient

load_dotenv()


# Configure module-level logging for indexing run visibility.
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s'
                    )
logger = logging.getLogger("indexer")

def index_documents():
    # Read blob source configuration from environment.
    account_name = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "adlscsuplatformdev")
    container_name = os.getenv("AZURE_BLOB_CONTAINER", "csudataplatform")
    blob_prefix = os.getenv("AZURE_BLOB_PREFIX", "staging/adhoc/Gaurav")
    account_url = f"https://{account_name}.blob.core.windows.net"
    logger.info(f"Blob source configured: account={account_name}, container={container_name}, prefix={blob_prefix}")
    
    # Create Azure credentials/clients for Blob and Azure OpenAI usage.
    token_provider = get_bearer_token_provider(DefaultAzureCredential(), os.getenv("TOKEN_PROVIDER"))
    blob_service_client = BlobServiceClient(account_url=account_url, credential=DefaultAzureCredential())
    container_client = blob_service_client.get_container_client(container_name)
    
    # Create embeddings client used by AzureSearch vector store.
    embeddings = AzureOpenAIEmbeddings(
        azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT"), 
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_ad_token_provider = token_provider) 
    
    # Initialize Azure Search vector store
    vector_store = AzureSearch(azure_search_endpoint = os.getenv("AZURE_SEARCH_ENDPOINT"), 
                               azure_search_key = os.getenv("AZURE_SEARCH_API_KEY"), 
                               index_name = os.getenv("AZURE_SEARCH_INDEX_NAME"),
                               embedding_function = embeddings.embed_query
                                   )
    

    # find PDF blobs
    pdf_blobs = [
        blob.name for blob in container_client.list_blobs(name_starts_with=blob_prefix.rstrip("/") + "/")
        if blob.name.lower().endswith(".pdf")
    ]
    if not pdf_blobs:
        logger.info(f"No PDF blobs found in container={container_name}, prefix={blob_prefix}")
        logger.warning("No PDF files found in configured blob path.")
        return
    logger.info(f"Found {len(pdf_blobs)} PDF blob(s). Starting indexing process...")


    all_splits = []
    for blob_name in pdf_blobs:
        # Each blob is downloaded to a temp file because PyPDFLoader reads file paths.
        try:
            logger.info(f"Downloading and processing blob: {blob_name}")
            blob_client = container_client.get_blob_client(blob_name)
            blob_data = blob_client.download_blob().readall()

            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
                temp_pdf.write(blob_data)
                temp_pdf_path = temp_pdf.name

            loader = PyPDFLoader(temp_pdf_path)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(documents)
            
            # Attach source metadata for traceability during retrieval/audits.
            for split in split_docs:
                split.metadata["source"] = Path(blob_name).name
                split.metadata["blob_name"] = blob_name
            
            all_splits.extend(split_docs)
            logger.info(f"Finished processing {blob_name}. Total splits so far: {len(all_splits)}")

            # Clean up temporary PDF after processing.
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
        except Exception as e:
            logger.error(f"Error processing {blob_name}: {e}")

    # Upload all chunks to Azure Search; embeddings are generated using embedding_function.
    # Text and vectors are stored together in the target index.

    if all_splits:
        try:
            vector_store.add_documents(all_splits)
            logger.info(f"Successfully indexed {len(all_splits)} document splits into Azure Search.")
        except Exception as e:
            logger.error(f"Error indexing documents to Azure Search: {e}")


if __name__ == "__main__":
    index_documents()                   




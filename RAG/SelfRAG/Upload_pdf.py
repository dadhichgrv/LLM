import os
from pathlib import Path
from dotenv import load_dotenv
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient


load_dotenv()

ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME", "adlscsuplatformdev")
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "csudataplatform")
BLOB_PREFIX = os.getenv("AZURE_BLOB_PREFIX", "staging/adhoc/Gaurav")
LOCAL_PDF_DIR = Path(os.getenv("LOCAL_PDF_DIR", "backend/data"))


def upload_pdfs() -> None:
    account_url = f"https://{ACCOUNT_NAME}.blob.core.windows.net"
    print("=== Blob Upload Configuration ===")
    print(f"Account URL      : {account_url}")
    print(f"Container Name   : {CONTAINER_NAME}")
    print(f"Blob Prefix(Path): {BLOB_PREFIX}")
    print(f"Local Folder     : {LOCAL_PDF_DIR.resolve()}")
    print("================================")

    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(account_url=account_url, credential=credential)


    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    if not LOCAL_PDF_DIR.exists():
        raise FileNotFoundError(f"Local PDF directory not found: {LOCAL_PDF_DIR.resolve()}")

    pdf_files = sorted(LOCAL_PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in: {LOCAL_PDF_DIR.resolve()}")
        return

    print(f"Uploading {len(pdf_files)} PDF file(s) to {CONTAINER_NAME}/{BLOB_PREFIX} ...")
    for pdf_path in pdf_files:
        blob_name = f"{BLOB_PREFIX.rstrip('/')}/{pdf_path.name}"
        print(f"Preparing upload -> container: {CONTAINER_NAME}, folder: {BLOB_PREFIX}, blob: {blob_name}")
        try:
            with open(pdf_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
            print(f"Uploaded: {pdf_path.name} -> {blob_name}")
        except Exception as e:
            print("Upload failed with the following context:")
            print(f"  Local File      : {pdf_path.resolve()}")
            print(f"  Account URL     : {account_url}")
            print(f"  Container Name  : {CONTAINER_NAME}")
            print(f"  Blob Folder Path: {BLOB_PREFIX}")
            print(f"  Target Blob Name: {blob_name}")
            print(f"  Error           : {e}")
            raise

    print("Upload completed successfully.")


if __name__ == "__main__":
    upload_pdfs()

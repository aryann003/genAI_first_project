import os

from pathlib import Path
from dotenv import load_dotenv

env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(env_path, override=True)

from langchain_community.document_loaders import WebBaseLoader

url = "https://www.example.com/"

loader = WebBaseLoader(url)
docs = loader.load()

print(len(docs))
print(docs[0].page_content)
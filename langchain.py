import os
import yaml
from datetime import datetime

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

CHUNK_SIZE = 1_000
CHUNK_OVERLAP = 200

def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Erreur de lecture YAML : {e}")
            return None

def resolve_env(value):
    if isinstance(value, str) and value.startswith("__ENV:"):
        return os.getenv(value[6:], "MISSING_ENV")
    return value

config = read_config("secrets/config.yaml")

embedder = AzureOpenAIEmbeddings(
    azure_endpoint=resolve_env(config["embedding"]["azure_endpoint"]),
    azure_deployment=resolve_env(config["embedding"]["azure_deployment"]),
    openai_api_version=resolve_env(config["embedding"]["azure_api_version"]),
    api_key=resolve_env(config["embedding"]["azure_api_key"])
)

vector_store = InMemoryVectorStore(embedder)

llm = AzureChatOpenAI(
    azure_endpoint=resolve_env(config["chat"]["azure_endpoint"]),
    azure_deployment=resolve_env(config["chat"]["azure_deployment"]),
    openai_api_version=resolve_env(config["chat"]["azure_api_version"]),
    api_key=resolve_env(config["chat"]["azure_api_key"]),
)

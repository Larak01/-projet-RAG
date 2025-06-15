import os
import yaml
from datetime import datetime

from llama_index.core import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from llama_index.core.vector_stores import SimpleVectorStore, VectorStoreQuery
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.readers.file import PyMuPDFReader

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

llm = AzureOpenAI(
    model=resolve_env(config["chat"]["azure_deployment"]),
    deployment_name=resolve_env(config["chat"]["azure_deployment"]),
    api_key=resolve_env(config["chat"]["azure_api_key"]),
    azure_endpoint=resolve_env(config["chat"]["azure_endpoint"]),
    api_version=resolve_env(config["chat"]["azure_api_version"]),
)

embedder = AzureOpenAIEmbedding(
    model=resolve_env(config["embedding"]["azure_deployment"]),
    deployment_name=resolve_env(config["embedding"]["azure_deployment"]),
    api_key=resolve_env(config["embedding"]["azure_api_key"]),
    azure_endpoint=resolve_env(config["embedding"]["azure_endpoint"]),
    api_version=resolve_env(config["embedding"]["azure_api_version"]),
)

Settings.llm = llm
Settings.embed_model = embedder

vector_store = SimpleVectorStore()

# Le reste du fichier reste inchangé (store_pdf_file, retrieve, answer_question...)

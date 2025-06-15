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

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)
    for split in all_splits:
        split.metadata = {
            'document_name': doc_name,
            'insert_date': datetime.now()
        }
    _ = vector_store.add_documents(documents=all_splits)
    return

def retrieve(question: str, top_k: int = 5):
    return vector_store.similarity_search(question, k=top_k)

def build_qa_messages(question: str, context: str, language: str) -> list[str]:
    messages = [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"""Use the following pieces of retrieved context to answer the question.
Answer in {language}.
If you don't know the answer, just say that you don't know.
Use three sentences maximum and keep the answer concise.

{context}"""),
        ("user", question)
    ]
    return messages

def answer_question(question: str, language: str = "English", top_k: int = 5) -> str:
    docs = retrieve(question, top_k=top_k)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, docs_content, language)
    response = llm.invoke(messages)
    return response.content

__all__ = ["answer_question", "store_pdf_file"]

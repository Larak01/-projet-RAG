from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import os

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ❗ Remplir ici directement les valeurs, remplace les chaînes par les vraies valeurs
EMBEDDING_CONFIG = {
    "model": "text-embedding-ada-002",
    "deployment_name": "text-embedding-ada-002",
    "api_key": "sk-...",
    "azure_endpoint": "https://mon-instance.openai.azure.com",
    "api_version": "2024-02-15",
}
CHAT_CONFIG = {
    "model": "gpt-35-turbo",
    "deployment_name": "gpt-35-turbo",
    "api_key": "sk-...",
    "azure_endpoint": "https://mon-instance.openai.azure.com",
    "api_version": "2024-02-15",
}

embedder = AzureOpenAIEmbeddings(**EMBEDDING_CONFIG)
llm = AzureChatOpenAI(**CHAT_CONFIG)

# Init vecteur vide
vector_store = FAISS.from_documents([Document(page_content="init")], embedder)
vector_store.delete(vector_store.index_to_docstore_id.keys())

def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        chunk.metadata.update({
            "document_name": doc_name,
            "insert_date": datetime.now().isoformat()
        })
    vector_store.add_documents(chunks)

def retrieve(question: str, k: int = 5):
    return vector_store.similarity_search(question, k=k)

def build_qa_messages(question: str, context: str, langue: str) -> list:
    instructions = {
        "Français": "Réponds en français.",
        "Anglais": "Answer in English.",
        "Espagnol": "Responde en español.",
        "Japonais": "日本語で答えてください。"
    }
    return [
        ("system", "You are a helpful assistant."),
        ("system", instructions.get(langue, "Answer in English.") + "\nUse max 3 sentences.\n" + context),
        ("user", question)
    ]

def answer_question(question: str, langue: str = "Français", k: int = 5) -> str:
    docs = retrieve(question, k=k)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context, langue)
    return llm.invoke(messages).content

from datetime import datetime
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
import streamlit as st

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embedder = AzureOpenAIEmbeddings(**st.secrets["embedding"])
llm = AzureChatOpenAI(**st.secrets["chat"])
vector_store = FAISS.from_documents([], embedder)

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    splits = splitter.split_documents(docs)
    for split in splits:
        split.metadata = {'document_name': doc_name, 'insert_date': datetime.now()}
    vector_store.add_documents(splits)

def retrieve(question: str, k: int = 5):
    return vector_store.similarity_search(question, k=k)

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"Use the context below to answer. Be brief.\n{context}"),
        ("user", question)
    ]

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question, k)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages).content

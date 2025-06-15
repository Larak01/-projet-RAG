import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import TextLoader, PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

import yaml

def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            return yaml.safe_load(file)
        except yaml.YAMLError as e:
            print(f"Erreur YAML: {e}")
            return None

config = read_config("secrets/config.yaml")


CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

embedder = AzureOpenAIEmbeddings(**config["embedding"])
llm = AzureChatOpenAI(**config["chat"])

# Initialiser un vecteur vide par défaut
vector_store = FAISS.from_documents([Document(page_content="init", metadata={"document_name": "placeholder", "insert_date": datetime.now()})], embedder)
vector_store.delete(vector_store.index_to_docstore_id.keys())  # purger le placeholder

def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract the following metadata from the content:
        - title, author, source, type, language, themes
        <content>\n{extract}\n</content>""")
    ]
    return llm.invoke(messages).content

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool=True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = text_splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {"document_name": doc_name, "insert_date": datetime.now()}

    if use_meta_doc and all_splits:
        extract = '\n\n'.join([split.page_content for split in all_splits[:10]])
        meta_doc = Document(page_content=get_meta_doc(extract), metadata={"document_name": doc_name, "insert_date": datetime.now()})
        all_splits.append(meta_doc)

    if all_splits:
        vector_store.add_documents(all_splits)

    return

def retrieve(question: str):
    return vector_store.similarity_search(question)

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are an assistant for question-answering tasks."),
        ("system", f"Use the following context to answer. Be concise.\n{context}"),
        ("user", question)
    ]

def answer_question(question: str) -> str:
    docs = retrieve(question)
    docs_content = "\n\n".join(doc.page_content for doc in docs)
    response = llm.invoke(build_qa_messages(question, docs_content))
    return response.content

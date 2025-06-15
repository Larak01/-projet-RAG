import streamlit as st
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Configuration avec st.secrets
embedder = AzureOpenAIEmbeddings(
    azure_endpoint=st.secrets["embedding"]["azure_endpoint"],
    azure_deployment=st.secrets["embedding"]["azure_deployment"],
    openai_api_version=st.secrets["embedding"]["azure_api_version"],
    api_key=st.secrets["embedding"]["azure_api_key"]
)

llm = AzureChatOpenAI(
    azure_endpoint=st.secrets["chat"]["azure_endpoint"],
    azure_deployment=st.secrets["chat"]["azure_deployment"],
    openai_api_version=st.secrets["chat"]["azure_api_version"],
    api_key=st.secrets["chat"]["azure_api_key"]
)

# Initialisation d'un store vide
vector_store = FAISS.from_documents([], embedder)

def get_meta_doc(extract: str) -> str:
    messages = [
        ("system", "You are a librarian extracting metadata from documents."),
        ("user", f"""Extract metadata:
- title
- author
- source
- type (e.g. scientific paper, literature)
- language
- themes (as keywords)
Return 'unknown' if not found.

<content>
{extract}
</content>""")
    ]
    return llm.invoke(messages).content

def store_pdf_file(file_path: str, doc_name: str, use_meta_doc: bool = True):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    all_splits = splitter.split_documents(docs)

    for split in all_splits:
        split.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    if use_meta_doc:
        extract = '\n\n'.join([split.page_content for split in all_splits[:min(10, len(all_splits))]])
        meta_doc = Document(
            page_content=get_meta_doc(extract),
            metadata={"document_name": doc_name, "insert_date": datetime.now()}
        )
        all_splits.append(meta_doc)

    vector_store.add_documents(all_splits)

def retrieve(question: str):
    return vector_store.similarity_search(question)

def build_qa_messages(question: str, context: str) -> list:
    return [
        ("system", "You are a helpful assistant for question-answering."),
        ("system", f"""Use the following context to answer briefly (3 sentences max). Say "I don't know" if unsure.\n{context}"""),
        ("user", question)
    ]

def answer_question(question: str, k: int = 5) -> str:
    docs = retrieve(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    messages = build_qa_messages(question, context)
    return llm.invoke(messages).content

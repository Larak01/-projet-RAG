import streamlit as st
import os
import sqlite3
from datetime import datetime

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings.fake import FakeEmbeddings
from langchain.llms.fake import FakeListLLM
from langchain_core.documents import Document

# 📁 Dossier pour les fichiers
os.makedirs("uploaded_docs", exist_ok=True)

# 🧠 Simule les embeddings + LLM
embedder = FakeEmbeddings()
llm = FakeListLLM(responses=["Ceci est une réponse factice."])
vector_store = FAISS.from_documents([], embedder)

# 🔢 Paramètres
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 🧩 Langue de réponse
langue_map = {
    "Français": "Réponds en français.",
    "Anglais": "Respond in English.",
    "Espagnol": "Responde en español.",
    "Japonais": "日本語で答えてください。"
}

# 🧠 Indexation PDF
def store_pdf_file(file_path: str, doc_name: str):
    loader = PyMuPDFLoader(file_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks = splitter.split_documents(docs)

    for chunk in chunks:
        chunk.metadata = {
            "document_name": doc_name,
            "insert_date": datetime.now()
        }

    vector_store.add_documents(chunks)

# 🔍 Recherche
def answer_question(question: str, langue: str):
    context_docs = vector_store.similarity_search(question, k=5)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    prompt = f"{langue_map[langue]}\n\nContexte:\n{context}\n\nQuestion: {question}"
    return llm.invoke(prompt)

# 🖼️ UI
st.title("📚 Assistant RAG - Projet MAG3")
langue = st.selectbox("Langue de la réponse :", list(langue_map.keys()))
uploaded_file = st.file_uploader("Déposez un fichier PDF", type=["pdf"])

if uploaded_file:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("Fichier chargé.")
    store_pdf_file(file_path, uploaded_file.name)

question = st.text_input("Posez votre question :")
if question:
    with st.spinner("Réflexion en cours..."):
        reponse = answer_question(question, langue)
        st.markdown("### Réponse :")
        st.write(reponse)

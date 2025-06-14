import streamlit as st
import os
from langchain import store_pdf_file as lc_store, answer_question as lc_answer
from llamaindex import store_pdf_file as li_store, answer_question as li_answer

# Titre
st.title("📚 Assistant Documentaire - Projet RAG")

# Choix du framework
framework = st.radio("Choisir le moteur d'indexation :", ["LangChain", "LlamaIndex"])

# Upload fichier PDF
uploaded_file = st.file_uploader("Déposer un fichier PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    os.makedirs("uploaded_docs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"Fichier {uploaded_file.name} chargé avec succès !")

    # Indexation du document
    if framework == "LangChain":
        lc_store(file_path, uploaded_file.name)
    else:
        li_store(file_path, uploaded_file.name)

# Entrée question utilisateur
question = st.text_input("Posez votre question sur le document :")

if question:
    with st.spinner("Recherche de la réponse..."):
        if framework == "LangChain":
            response = lc_answer(question)
        else:
            response = li_answer(question)

        st.markdown("### Réponse :")
        st.write(response)

# Pied de page
st.markdown("---")
st.caption("Projet MAG 3 — Hands-on RAG — Larak01")

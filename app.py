import streamlit as st
from datetime import datetime
import sqlite3
from rag_langchain import store_pdf_file, answer_question

os.makedirs("uploaded_docs", exist_ok=True)
st.set_page_config(page_title="Assistant RAG", layout="centered")
st.title("📚 Assistant Documentaire - Projet RAG")

# Paramètres
with st.sidebar:
    st.header("Paramètres")
    langue = st.selectbox("Langue :", ["Français", "Anglais", "Espagnol", "Japonais"])
    top_k = st.slider("Top K documents :", 1, 10, 5)

# PDF Upload
uploaded = st.file_uploader("📄 Déposez un fichier PDF", type="pdf")
if uploaded:
    path = f"uploaded_docs/{uploaded.name}"
    with open(path, "wb") as f:
        f.write(uploaded.read())
    st.success("Fichier chargé.")
    try:
        store_pdf_file(path, uploaded.name)
        st.success("Indexation terminée.")
    except Exception as e:
        st.error(f"Erreur indexation : {e}")

# Question
question = st.text_input("❓ Posez une question sur le document :")
if question:
    with st.spinner("Recherche..."):
        reponse = answer_question(question, langue, top_k)
        st.markdown("### Réponse :")
        st.write(reponse)

        # Feedback
        feedback = st.radio("Utile ?", ["Oui", "Non"], horizontal=True)
        if feedback:
            conn = sqlite3.connect("feedback.db")
            c = conn.cursor()
            c.execute("CREATE TABLE IF NOT EXISTS feedbacks (timestamp TEXT, question TEXT, response TEXT, feedback TEXT)")
            c.execute("INSERT INTO feedbacks VALUES (?, ?, ?, ?)",
                      (datetime.now().isoformat(), question, reponse, feedback))
            conn.commit()
            conn.close()
            st.success("Merci pour votre retour !")

st.caption("Projet RAG - MAG3 2025")

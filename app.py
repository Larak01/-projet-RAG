import streamlit as st
from datetime import datetime
import sqlite3
from rag_langchain import store_pdf_file as lc_store, answer_question as lc_answer
from llamaindex import store_pdf_file as li_store, answer_question as li_answer

st.title("📚 Assistant RAG – Projet MAG3")

framework = st.radio("Choix du moteur :", ["LangChain", "LlamaIndex"])
langue = st.selectbox("Langue de réponse :", ["Français", "Anglais", "Espagnol", "Japonais"])
top_k = st.slider("Top-K documents :", 1, 10, 5)

conn = sqlite3.connect("feedback.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS feedbacks (timestamp TEXT, question TEXT, response TEXT, feedback TEXT)")

langue_map = {
    "Français": "Réponds en français.",
    "Anglais": "Answer in English.",
    "Espagnol": "Responde en español.",
    "Japonais": "日本語で答えてください。"
}

st.markdown("### 📄 Charger un PDF")
uploaded_file = st.file_uploader("Uploader un fichier PDF", type=["pdf"])
if uploaded_file:
    path = f"uploaded_docs/{uploaded_file.name}"
    with open(path, "wb") as f:
        f.write(uploaded_file.read())
    st.success("✅ Fichier chargé.")
    if framework == "LangChain":
        lc_store(path, uploaded_file.name)
    else:
        li_store(path, uploaded_file.name)
    st.success("✅ Indexation terminée.")

question = st.text_input("❓ Votre question :")
if question:
    full_q = langue_map[langue] + "\n" + question
    with st.spinner("Recherche..."):
        response = lc_answer(full_q, k=top_k) if framework == "LangChain" else li_answer(full_q, k=top_k)
        st.markdown("### Réponse :")
        st.write(response)

    feedback = st.radio("Utile ?", ["Oui", "Non"], horizontal=True)
    if feedback:
        c.execute("INSERT INTO feedbacks VALUES (?, ?, ?, ?)",
                  (datetime.now().isoformat(), question, response, feedback))
        conn.commit()
        st.success("Merci pour votre retour !")

conn.close()

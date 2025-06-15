import streamlit as st
import os
import sqlite3
from datetime import datetime
from langchain import store_pdf_file as lc_store, answer_question as lc_answer
from llamaindex import store_pdf_file as li_store, answer_question as li_answer

# ✅ Créer le dossier s'il n'existe pas
os.makedirs("uploaded_docs", exist_ok=True)

st.title("📚 Assistant Documentaire - Projet RAG")

# Choix du framework
framework = st.radio("Choisir le moteur d'indexation :", ["LangChain", "LlamaIndex"])

# Choix de la langue de réponse
langue = st.selectbox("Langue de la réponse :", ["Français", "Anglais", "Espagnol", "Japonais"])

# Nombre de documents à récupérer (top_k)
top_k = st.slider("Nombre de documents similaires à récupérer :", min_value=1, max_value=10, value=5)

# Connexion SQLite
conn = sqlite3.connect("feedback.db")
c = conn.cursor()
c.execute("CREATE TABLE IF NOT EXISTS feedbacks (timestamp TEXT, question TEXT, response TEXT, feedback TEXT)")

@st.cache_data
def cached_store(framework, path, name):
    if framework == "LangChain":
        lc_store(path, name)
    else:
        li_store(path, name)

# 📥 Upload PDF
uploaded_file = st.file_uploader("Déposer un fichier PDF", type=["pdf"])

if uploaded_file is not None:
    file_path = os.path.join("uploaded_docs", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    st.success(f"✅ Fichier {uploaded_file.name} chargé avec succès !")

    # Indexation avec gestion d'erreur
    st.write("📥 Indexation en cours...")
    try:
        cached_store(framework, file_path, uploaded_file.name)
        st.success("Indexation réussie.")
    except Exception as e:
        st.error(f"Erreur d’indexation : {e}")

# 💬 Question utilisateur
question = st.text_input("Posez votre question sur le document :")

# Mapping des instructions de langue
langue_map = {
    "Français": "Réponds en français.",
    "Anglais": "Respond in English.",
    "Espagnol": "Responde en español.",
    "Japonais": "日本語で答えてください。"
}

response = ""

if question:
    with st.spinner("Recherche de la réponse..."):
        full_question = langue_map.get(langue, "") + "\n" + question
        if framework == "LangChain":
            response = lc_answer(full_question, k=top_k)
        else:
            response = li_answer(full_question, k=top_k)
        st.markdown("### Réponse :")
        st.write(response)

# ✅ Feedback utilisateur
st.markdown("### Votre avis sur la réponse :")
feedback = st.radio("Cette réponse était-elle utile ?", ["Oui", "Non"], horizontal=True)

if feedback and question and response:
    timestamp = datetime.now().isoformat()
    c.execute("INSERT INTO feedbacks VALUES (?, ?, ?, ?)",
              (timestamp, question, response, feedback))
    conn.commit()
    st.success("Merci pour votre retour !")

conn.close()
st.markdown("---")
st.caption("Projet MAG 3 — Hands-on RAG — Larak01")

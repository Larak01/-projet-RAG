import streamlit as st
import sqlite3
import pandas as pd
from langchain import answer_question as answer_lc, store_pdf_file as store_lc
from llamaindex import answer_question as answer_ll, store_pdf_file as store_ll
import tempfile
from datetime import datetime

# --- UI Setup ---
st.set_page_config(page_title="Projet RAG", layout="centered")
st.title("📚 RAG : Retrieval-Augmented Generation")

# --- Choix du framework ---
framework = st.radio("Choisissez le moteur RAG", ["LangChain", "LlamaIndex"], horizontal=True)

# --- Choix de la langue ---
langue = st.selectbox("Choisissez la langue de réponse", ["Français", "Anglais", "Espagnol", "Japonais"])
lang_codes = {
    "Français": "French",
    "Anglais": "English",
    "Espagnol": "Spanish",
    "Japonais": "Japanese"
}
langue_cible = lang_codes[langue]

# --- Choix du top_k ---
top_k = st.slider("Nombre de documents similaires à récupérer", min_value=1, max_value=10, value=5)

# --- Upload de PDF ---
uploaded_file = st.file_uploader("Téléversez un fichier PDF", type="pdf")
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.info(f"Indexation du fichier : {uploaded_file.name} avec {framework}")
    if framework == "LangChain":
        store_lc(tmp_path, uploaded_file.name)
    else:
        store_ll(tmp_path, uploaded_file.name)
    st.success("Fichier indexé avec succès !")

# --- Zone de question ---
question_utilisateur = st.text_area("Posez votre question", "")

# --- Traitement ---
if st.button("Poser la question"):
    if question_utilisateur.strip() == "":
        st.warning("Veuillez entrer une question.")
    else:
        with st.spinner("Génération de la réponse..."):
            if framework == "LangChain":
                reponse = answer_lc(question_utilisateur, language=langue_cible, top_k=top_k)
            else:
                reponse = answer_ll(question_utilisateur, language=langue_cible, top_k=top_k)
        st.success("Réponse générée :")
        st.markdown(reponse)

        # --- Feedback utilisateur ---
        feedback = st.radio("Êtes-vous satisfait de la réponse ?", ["👍 Oui", "👎 Non"])
        if st.button("Enregistrer le feedback"):
            try:
                conn = sqlite3.connect("feedback.db")
                cursor = conn.cursor()
                cursor.execute('''CREATE TABLE IF NOT EXISTS feedbacks (
                                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                                    question TEXT,
                                    reponse TEXT,
                                    satisfaction TEXT,
                                    timestamp TEXT
                                )''')
                conn.commit()
                cursor.execute("INSERT INTO feedbacks (question, reponse, satisfaction, timestamp) VALUES (?, ?, ?, ?)",
                               (question_utilisateur, reponse, feedback, datetime.now().isoformat()))
                conn.commit()
                conn.close()
                st.success("Feedback enregistré avec succès ✅")
            except Exception as e:
                st.error("Erreur lors de l'enregistrement du feedback")
                st.code(str(e))

# --- Footer ---
st.markdown("---")
st.markdown("Projet RAG  — 2025")

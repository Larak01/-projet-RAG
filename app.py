import streamlit as st
import os
import sqlite3
from datetime import datetime
from langchain import store_pdf_file as lc_store, answer_question as lc_answer
from llamaindex import store_pdf_file as li_store, answer_question as li_answer

# Titre
st.title("📚 Assistant Documentaire - Projet RAG")

# Choix du framework
framework = st.radio("Choisir le moteur d'indexation :", ["LangChain", "LlamaIndex"])

# Choix de la langue de réponse
langue = st.selectbox("Langue de la réponse :", ["Français", "Anglais", "Espagnol", "Japonais"])

# Choix du nombre de documents à récupérer (top_k)
top_k = st.slider("Nombre de documents similaires à récupérer :", min_value=1, max_value=10, value=5)

# Connexion à la base SQLite
conn = sqlite3.connect("feedback.db")
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS feedbacks
             (timestamp TEXT, question TEXT, response TEXT, feedback TEXT)''')

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

# Mapping des langues vers codes ou instructions
langue_map = {
    "Français": "Réponds en français.",
    "Anglais": "Respond in English.",
    "Espagnol": "Responde en español.",
    "Japonais": "日本語で答えてください。"
}

if question:
    with st.spinner("Recherche de la réponse..."):
        message_prefix = langue_map.get(langue, "") + "\n"
        full_question = message_prefix + question

        if framework == "LangChain":
            response = lc_answer(full_question, k=top_k)
        else:
            response = li_answer(full_question, k=top_k)

        st.markdown("### Réponse :")
        st.write(response)

        # Feedback utilisateur avec enregistrement dans SQLite
        st.markdown("### Votre avis sur la réponse :")
        feedback = st.feedback("Cette réponse était-elle utile ?", key="feedback")

        if feedback:
            timestamp = datetime.now().isoformat()
            c.execute("INSERT INTO feedbacks VALUES (?, ?, ?, ?)",
                      (timestamp, question, response, feedback))
            conn.commit()
            st.success("Merci pour votre retour !")

# Fermeture base
conn.close()

# Pied de page
st.markdown("---")
st.caption("Projet MAG 3 — Hands-on RAG — Larak01")

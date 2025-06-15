# ✅ Final version of the main application file with all required updates

import os
import tempfile
import streamlit as st
import pandas as pd
import sqlite3
import glob

from rag.langchain import answer_question as answer_langchain
from rag.llamaindex import answer_question as answer_llamaindex
from rag.langchain import store_pdf_file as store_pdf_langchain
from rag.llamaindex import store_pdf_file as store_pdf_llamaindex
from rag.langchain import delete_file_from_store as delete_file_langchain
from rag.llamaindex import delete_file_from_store as delete_file_llamaindex

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="📄",
)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

# SQLite DB initialization
conn = sqlite3.connect("feedback.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        question TEXT,
        answer TEXT,
        rating TEXT
    )
''')
conn.commit()

def load_sample_pdfs(framework, samples_dir="samples"):
    sample_paths = glob.glob(f"{samples_dir}/*.pdf")
    for path in sample_paths:
        file_name = os.path.basename(path)
        if file_name not in st.session_state['stored_files']:
            if framework == "LangChain":
                store_pdf_langchain(path, file_name)
            else:
                store_pdf_llamaindex(path, file_name)
            st.session_state['stored_files'].append(file_name)

def main():
    st.title("🧠 Analyse de documents enrichie")
    st.subheader("Chargez vos PDF, interrogez-les, et laissez un avis.")

    # Langue de réponse
    lang = st.selectbox("🌐 Choisissez la langue de réponse", ["Français", "Anglais", "Espagnol", "Japonais"])

    # Choix du framework
    framework = st.radio("🧰 Choisissez le framework d'indexation", ["LangChain", "LlamaIndex"])

    # Nombre de documents à récupérer
    k = st.slider("📄 Nombre de documents à considérer pour la réponse", min_value=1, max_value=10, value=5)

    # Charger les articles PDF de base
    load_sample_pdfs(framework)

    uploaded_files = st.file_uploader("📂 Déposez vos fichiers ici", type=['pdf'], accept_multiple_files=True)

    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({"Nom du fichier": f.name, "Taille (KB)": f"{size_in_kb:.2f}"})
            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, f.name)
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                if framework == "LangChain":
                    store_pdf_langchain(path, f.name)
                else:
                    store_pdf_llamaindex(path, f.name)
                st.session_state['stored_files'].append(f.name)

        st.table(pd.DataFrame(file_info))

    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        if framework == "LangChain":
            delete_file_langchain(name)
        else:
            delete_file_llamaindex(name)

    question = st.text_input("❓ Posez votre question")

    if st.button("Analyser") and question:
        if framework == "LangChain":
            response = answer_langchain(question)
        else:
            response = answer_llamaindex(question)

        st.text_area("🧾 Réponse générée", value=response, height=200)

        rating = st.radio("📝 Évaluez cette réponse", ["👍 Pertinente", "👎 Peu utile"])
        if st.button("Soumettre l'avis"):
            cursor.execute("INSERT INTO feedback (question, answer, rating) VALUES (?, ?, ?)", (question, response, rating))
            conn.commit()
            st.success("Merci pour votre retour !")
    else:
        st.text_area("🧾 Réponse générée", value="", height=200)

if __name__ == "__main__":
    main()

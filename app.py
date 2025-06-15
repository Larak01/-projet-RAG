import os
import tempfile
import streamlit as st
import pandas as pd

from rag.langchain import answer_question, delete_file_from_store, store_pdf_file

st.set_page_config(
    page_title="Analyse de documents",
    page_icon="📄",
)

if 'stored_files' not in st.session_state:
    st.session_state['stored_files'] = []

def main():
    st.title("🧠 Analyse de documents")
    st.subheader("Chargez vos documents PDF et interrogez-les avec l'IA.")

    uploaded_files = st.file_uploader(
        label="📂 Déposez vos fichiers ici",
        type=['pdf'],
        accept_multiple_files=True
    )

    file_info = []
    if uploaded_files:
        for f in uploaded_files:
            size_in_kb = len(f.getvalue()) / 1024
            file_info.append({
                "Nom du fichier": f.name,
                "Taille (KB)": f"{size_in_kb:.2f}"
            })

            if f.name.endswith('.pdf') and f.name not in st.session_state['stored_files']:
                temp_dir = tempfile.mkdtemp()
                path = os.path.join(temp_dir, "temp.pdf")
                with open(path, "wb") as outfile:
                    outfile.write(f.read())
                store_pdf_file(path, f.name)
                st.session_state['stored_files'].append(f.name)

        df = pd.DataFrame(file_info)
        st.table(df)

    files_to_be_deleted = set(st.session_state['stored_files']) - {f['Nom du fichier'] for f in file_info}
    for name in files_to_be_deleted:
        st.session_state['stored_files'].remove(name)
        delete_file_from_store(name)

    question = st.text_input("❓ Posez votre question")

    if st.button("Analyser") and question:
        model_response = answer_question(question)
        st.text_area("Réponse générée", value=model_response, height=200)
    else:
        st.text_area("Réponse générée", value="", height=200)

if __name__ == "__main__":
    main()

import streamlit as st
import os
from datetime import datetime

st.set_page_config(page_title="Test RAG", page_icon="📘")
st.title("✅ Déploiement Streamlit réussi")

st.markdown("Cette app est un test minimal pour vérifier que Streamlit Cloud fonctionne.")

st.subheader("📤 Uploader un PDF (non traité)")
uploaded_file = st.file_uploader("Choisissez un fichier PDF", type=["pdf"])

if uploaded_file:
    os.makedirs("uploaded_docs", exist_ok=True)
    with open(os.path.join("uploaded_docs", uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    st.success(f"✅ Fichier reçu : {uploaded_file.name}")

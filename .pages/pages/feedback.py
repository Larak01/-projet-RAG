import streamlit as st
import pandas as pd
import sqlite3

st.set_page_config(
    page_title="Retour utilisateur",
    page_icon="🗳️"
)

st.title("📊 Visualisation des retours utilisateurs")

# Connexion à la base
conn = sqlite3.connect("feedback.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("SELECT question, answer, rating FROM feedback")
data = cursor.fetchall()

df = pd.DataFrame(data, columns=["Question", "Réponse", "Évaluation"])

if df.empty:
    st.info("Aucun feedback collecté pour l'instant.")
else:
    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistiques")
    st.bar_chart(df["Évaluation"].value_counts())

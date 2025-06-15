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
    st.subheader("🔎 Filtrer par évaluation")
    filtre = st.radio("Choisissez un type d'avis", ["Tous", "👍 Pertinente", "👎 Peu utile"])

    if filtre != "Tous":
        df = df[df["Évaluation"] == filtre]

    st.dataframe(df, use_container_width=True)

    st.subheader("📈 Statistiques")
    stats = df["Évaluation"].value_counts().rename_axis("Évaluation").reset_index(name="Nombre")
    st.bar_chart(stats.set_index("Évaluation"))

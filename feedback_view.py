import sqlite3
import pandas as pd
import streamlit as st

# Connexion à la base de données SQLite
try:
    conn = sqlite3.connect('feedback.db')
    df = pd.read_sql_query(
        "SELECT * FROM feedbacks ORDER BY timestamp DESC", conn
    )
    conn.close()

    st.title("📝 Feedback utilisateur")

    # Filtrage par note
    note_filter = st.selectbox("Filtrer par satisfaction", ["Toutes", "👍", "👎"])
    if note_filter != "Toutes":
        df = df[df["note"] == note_filter]

    # Affichage des données filtrées
    st.dataframe(df)

    # Bouton d'export CSV
    st.download_button(
        label="📥 Exporter les feedbacks en CSV",
        data=df.to_csv(index=False),
        file_name="feedbacks.csv",
        mime="text/csv"
    )

except Exception as e:
    st.error("Erreur lors de la lecture de la base de données :")
    st.code(str(e))

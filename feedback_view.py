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
    st.dataframe(df)

except Exception as e:
    st.error("Erreur lors de la lecture de la base de données :")
    st.code(str(e))

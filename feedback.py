import sqlite3
import pandas as pd
import streamlit as st

st.title("📊 Feedback utilisateurs")

conn = sqlite3.connect("feedback.db", check_same_thread=False)
cursor = conn.cursor()

# 🔧 Assure que la table existe
cursor.execute('''
    CREATE TABLE IF NOT EXISTS feedback (
        question TEXT,
        answer TEXT,
        rating TEXT
    )
''')
conn.commit()

cursor.execute("SELECT question, answer, rating FROM feedback")
data = cursor.fetchall()

df = pd.DataFrame(data, columns=["Question", "Réponse", "Évaluation"])
st.dataframe(df)

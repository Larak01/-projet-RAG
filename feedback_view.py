import sqlite3
import pandas as pd
import streamlit as st

conn = sqlite3.connect('feedback.db')
df = pd.read_sql_query("SELECT * FROM feedbacks ORDER BY timestamp DESC", conn)
conn.close()

st.title("📝 Feedback utilisateur")
st.dataframe(df)

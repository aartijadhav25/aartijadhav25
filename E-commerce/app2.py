import pandas as pd
import numpy as np
import streamlit as st
import faiss
import sqlite3
import random
import requests
from sentence_transformers import SentenceTransformer  # or use HuggingFaceEmbeddings
import os

# --- Streamlit setup ---
st.set_page_config(page_title="Laptop Search", layout="wide")
st.title("💻 AI-Powered Laptop Search (Ollama Edition)")
# --- Load dataset ---
df = pd.read_csv(r"D:\new ecommerce\data_laptop_specification.csv")
df.fillna("", inplace=True)
battery_life = ["10 hours", "8 hours", "12 hours", "6 hours", "5 hours"]
df['battery_life'] = [random.choice(battery_life) for _ in range(len(df))]
df.to_csv(r"D:\new ecommerce\data_laptop_specification.csv", index=False)

# --- Full description ---
df = pd.read_csv(r"D:\new ecommerce\data_laptop_specification.csv")
df['full_description'] = ((df['title'].astype(str) + " " + df['TypeName'].astype(str) + " " +
                           df['cpu_name'].astype(str) + " " + df['gpu_name'].astype(str) + " " +
                           df['OpSys'].astype(str) + " " + df['review'].astype(str) + " " +
                           df['Company'].astype(str) + " " + df['Price'].astype(str) + " " +
                           df['Ram'].astype(str) + " " + df['battery_life'].astype(str)))
df['full_description'] = df['full_description'].str.replace(r'\s+', ' ', regex=True).str.strip()

# --- Embeddings ---
model = SentenceTransformer('all-MiniLM-L6-v2')  # Replaceable with huggingface embedding if needed
embeddings = model.encode(df['full_description'].tolist(), show_progress_bar=True)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(np.array(embeddings))

# --- SQLite logging ---
conn = sqlite3.connect("logs.db", check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS logs (query TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')
conn.commit()

def log_query(query):
    cursor.execute("INSERT INTO logs (query) VALUES (?)", (query,))
    conn.commit()

def fetch_history(limit=5):
    cursor.execute("SELECT query, timestamp FROM logs ORDER BY timestamp DESC LIMIT ?", (limit,))
    return cursor.fetchall()

# --- Search ---
def search_products(query, top_k=10):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    return df.iloc[indices[0]]

# --- Ollama Call ---
def ollama_ask(prompt, model="mistral"):
    url = "http://localhost:11434/api/generate"
    data = {
        "model": model,
        "prompt": prompt,
        "stream": False
    }
    try:
        response = requests.post(url, json=data)
        return response.json()["response"]
    except Exception as e:
        return f"Error contacting Ollama: {e}"

# --- UI Layout ---
left_col, right_col = st.columns([3, 1])

with left_col:
    st.subheader("🔎 Search for a Laptop")
    query = st.text_input("Enter your query", placeholder="e.g., i5 SSD laptop under ₹50000")

    if st.button("Search") and query.strip():
        log_query(query)
        results = search_products(query)

        st.subheader("Top 10 Matching Laptops")
        for i, (_, row) in enumerate(results.iterrows(), 1):
            desc = row["full_description"]
            prompt = f"You are a laptop recommender. Given the following specs:\n\n{desc}\n\nHow suitable is this for a user looking for: '{query}'?"
            ai_response = ollama_ask(prompt)

            st.markdown("----")
            st.markdown(f"### {i}. 🖥️ {row['Company']} - {row['title']}")
            st.markdown(f"*CPU*: {row['cpu_name']}  \n🎮 *GPU*: {row['gpu_name']}  \n *Price*: ₹{row['Price']}")
            st.markdown(f"**Ollama Says**: {ai_response}")

with right_col:
    st.subheader(" Chat History")
    if st.button(" Clear History"):
        cursor.execute("DELETE FROM logs")
        conn.commit()
        st.success("Search history cleared.")
    history = fetch_history()
    if history:
        for q, t in history:
            st.markdown(f"- {q}  \n<span style='font-size: 10px; color: gray;'>{t}</span>", unsafe_allow_html=True)
    else:
        st.write("No search history yet.")

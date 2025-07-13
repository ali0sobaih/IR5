import sqlite3
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os

from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  
DATA_DIR = BASE_DIR / "offline"

def process_tfidf(table_name):
    path = DATA_DIR / "ir_project.db" 
    conn = sqlite3.connect(path)
    df = pd.read_sql(f"SELECT doc_id, processed_doc FROM {table_name}", conn)
    conn.close()

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['processed_doc'])

    os.makedirs("offline_data", exist_ok=True)
    joblib.dump({
        "doc_ids": df['doc_id'].tolist(),
        "vectorizer": vectorizer,
        "matrix": tfidf_matrix
    }, f"offline_data/tfidf_{table_name}.joblib")

if __name__ == "__main__":
    process_tfidf("antique")
    process_tfidf("quora")

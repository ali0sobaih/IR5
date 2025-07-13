import sqlite3
import pandas as pd
import joblib
from rank_bm25 import BM25Okapi
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent  
DATA_DIR = BASE_DIR / "offline"


def process_bm25(table_name):
    path = DATA_DIR / "ir_project.db" 
    conn = sqlite3.connect(path)
    df = pd.read_sql(f"SELECT doc_id, processed_doc FROM {table_name}", conn)
    conn.close()

    tokenized = [doc.split() for doc in df["processed_doc"]]
    bm25 = BM25Okapi(tokenized)

    os.makedirs("offline_data", exist_ok=True)
    joblib.dump({
        "doc_ids": df["doc_id"].tolist(),
        "bm25": bm25,
        "tokenized_docs": tokenized
    }, f"offline_data/bm25_{table_name}.joblib")

if __name__ == "__main__":
    process_bm25("antique")
    process_bm25("quora")

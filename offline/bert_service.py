import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "offline"
FAISS_STORE = BASE_DIR / "faiss_store"

def process_bert(table_name):
    db_path = DATA_DIR / "ir_project.db" 
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT doc_id, doc FROM {table_name}", conn)
    conn.close()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["doc"].tolist(), convert_to_tensor=False)
    embeddings = np.array(embeddings).astype("float32")

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    store_path = FAISS_STORE / table_name
    os.makedirs(store_path, exist_ok=True)
    faiss.write_index(index, str(store_path / "index.faiss"))

if __name__ == "__main__":
    process_bert("antique")
    process_bert("quora")

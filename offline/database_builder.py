import pandas as pd
import sqlite3
import json
import os
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pathlib import Path
import nltk
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

BASE_DIR = Path(__file__).parent.parent  
DATA_DIR = BASE_DIR / "data"

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    ps = PorterStemmer()
    stems = [ps.stem(token) for token in filtered_tokens]
    return " ".join(stems)

def create_table(conn, name):
    conn.execute(f"DROP TABLE IF EXISTS {name}")
    conn.execute(f"""
        CREATE TABLE {name} (
            doc_id TEXT PRIMARY KEY,
            doc TEXT,
            processed_doc TEXT
        )
    """)

def insert_documents(conn, table, documents):
    for doc_id, text in documents.items():
        processed = preprocess(text)
        conn.execute(
            f"INSERT INTO {table} (doc_id, doc, processed_doc) VALUES (?, ?, ?)",
            (doc_id, text, processed)
        )
    conn.commit()

def load_antique():
    docs = {}
    antique_path = DATA_DIR / "antique" / "collection.txt"
    with open(antique_path, encoding="utf8") as f:
        for line in f:
            doc_id, text = line.strip().split("\t", 1)
            docs[doc_id] = text
    return docs

def load_quora():
    docs = {}
    quora_path = DATA_DIR / "quora" / "corpus.jsonl"
    with open(quora_path, encoding="utf8") as f:
        for line in f:
            item = json.loads(line)
            docs[item["_id"]] = item["text"]
    return docs

def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("STARTED")
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect("offline/ir_project.db")

    antique_docs = load_antique()
    quora_docs = load_quora()

    create_table(conn, "antique")
    insert_documents(conn, "antique", antique_docs)

    create_table(conn, "quora")
    insert_documents(conn, "quora", quora_docs)

    conn.close()

if __name__ == "__main__":
    main()

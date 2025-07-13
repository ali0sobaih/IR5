# dataPipline.py (Hardcoded Paths)

import sqlite3
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
from pathlib import Path
import joblib

from services.preprocessing_service import preprocess

DB_PATH = Path("C:/Users/Lenovo/Desktop/5th/second simester/IR/IR_PROJECT/offline/ir_project.db")
FAISS_STORE = Path("C:/Users/Lenovo/Desktop/5th/second simester/IR/IR_PROJECT/offline_data")

os.makedirs(FAISS_STORE, exist_ok=True)

DATABASE_TABLES = ["antique", "quora"] 

# --- Main Vocabulary Preparation Logic ---
def prepare_general_vocabulary_and_faiss_index():
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}. Please ensure 'ir_project.db' exists.")
        return

    all_document_texts = []
    print(f"Connecting to database: {DB_PATH}")

    try:
        conn = sqlite3.connect(DB_PATH)
        for table_name in DATABASE_TABLES:
            print(f"Reading documents from table: {table_name}...")
            df = pd.read_sql(f"SELECT doc FROM {table_name}", conn)
            all_document_texts.extend(df["doc"].tolist())
            print(f"  - Added {len(df)} documents from {table_name}.")
        conn.close()
        print(f"Total documents loaded from database: {len(all_document_texts)}")

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading documents: {e}")
        return

    if not all_document_texts:
        print("No documents loaded. Cannot create vocabulary.")
        return

    # --- Step 1: Preprocess all documents and collect unique words ---
    all_unique_words = set()
    print("Preprocessing all documents and collecting unique words...")
    for doc_text in all_document_texts:
        preprocessed_text = preprocess(doc_text)
        if preprocessed_text:
            for word in preprocessed_text.split():
                all_unique_words.add(word)

    vocabulary_words = sorted(list(all_unique_words))
    print(f"\nTotal unique words for vocabulary: {len(vocabulary_words)}")

    if not vocabulary_words:
        print("No unique words collected after preprocessing. Cannot create embeddings/FAISS index.")
        return

    # --- Step 2: Encode the vocabulary words using SentenceTransformer ---
    print("Loading SentenceTransformer model for encoding...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Encoding {len(vocabulary_words)} vocabulary words...")
    vocabulary_embeddings = model.encode(vocabulary_words, batch_size=64, convert_to_tensor=False)
    vocabulary_embeddings = np.array(vocabulary_embeddings).astype("float32")
    print("Encoding complete.")

    # --- Step 3: Build a FAISS index ---
    print("Building FAISS index...")
    dimension = vocabulary_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vocabulary_embeddings)
    print(f"FAISS index built with {index.ntotal} vectors.")

    # --- Step 4: Save the vocabulary and FAISS index ---
    VOCAB_FILE_PATH = FAISS_STORE / "general_semantic_vocabulary.joblib"
    FAISS_INDEX_PATH = FAISS_STORE / "general_semantic_vocabulary.faiss"

    print(f"Saving vocabulary to: {VOCAB_FILE_PATH}")
    joblib.dump(vocabulary_words, VOCAB_FILE_PATH)
    print(f"Saving FAISS index to: {FAISS_INDEX_PATH}")
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print("Vocabulary and FAISS index saved successfully!")

if __name__ == "__main__":
    prepare_general_vocabulary_and_faiss_index()
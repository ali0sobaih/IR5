# main.py (No longer performs indexing, only verifies data/FAISS paths)
from pathlib import Path
import os

print("main.py: This script no longer performs embedding or vector DB creation.")
print("It is assumed that process_bert.py has been run to create FAISS indexes.")

BASE_DIR = Path(__file__).parent.parent
FAISS_STORE = BASE_DIR / "faiss_store"
DATA_DIR = BASE_DIR / "data" # Changed from 'offline' to 'data' based on user input

# Check if necessary FAISS files exist
if not (FAISS_STORE / "antique" / "index.faiss").exists():
    print(f"Error: {FAISS_STORE / 'antique' / 'index.faiss'} not found. Please run process_bert.py first.")
if not (FAISS_STORE / "quora" / "index.faiss").exists():
    print(f"Error: {FAISS_STORE / 'quora' / 'index.faiss'} not found. Please run process_bert.py first.")

# Check if necessary data files exist
if not (DATA_DIR / "antique" / "collection.txt").exists():
    print(f"Error: {DATA_DIR / 'antique' / 'collection.txt'} not found. Ensure your data is in the correct location.")
if not (DATA_DIR / "quora" / "corpus.jsonl").exists():
    print(f"Error: {DATA_DIR / 'quora' / 'corpus.jsonl'} not found. Ensure your data is in the correct location.")

print("main.py: Verification complete. Proceed to run chat_api.py if FAISS indexes and data files are ready.")
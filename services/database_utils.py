# services/database_utils.py

import sqlite3
from pathlib import Path

# Ensure this path is correct relative to your project's root directory
DB_PATH = Path("offline/ir_project.db") 

def get_doc_text_by_id(dataset: str, doc_id: str) -> str:
    """
    Fetches the full text of a single document from the SQLite database
    for a given dataset and document ID.
    """
    conn = None
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        # Use a parameterized query to prevent SQL injection and handle special characters
        cursor.execute(f"SELECT doc FROM `{dataset}` WHERE doc_id = ?", (doc_id,))
        result = cursor.fetchone()
        if result:
            return result[0] # Return the document text
        else:
            # print(f"Warning: Document {doc_id} not found in dataset {dataset}.") # Uncomment for debugging
            return "" # Return empty string if document not found
    except sqlite3.OperationalError as e:
        print(f"SQLite error fetching doc {doc_id} from {dataset}: {e}")
        return ""
    finally:
        if conn:
            conn.close()
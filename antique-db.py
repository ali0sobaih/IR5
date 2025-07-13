import sqlite3
import pandas as pd

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect('ir_project.db')
cursor = conn.cursor()

# Create tables (same structure as your friend's)
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    source TEXT
)
""")    

cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    query_text TEXT,
    source TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS qrels (
    query_id TEXT,
    doc_id TEXT,
    relevance INTEGER,
    source TEXT,
    PRIMARY KEY (query_id, doc_id)
)
""")

# Load your ANTIQUE collection
antique_df = pd.read_csv('documents/antique/collection.txt', sep='\t', header=None, names=['doc_id', 'content'])

# Insert into SQLite (with 'antique' as source)
for _, row in antique_df.iterrows():
    cursor.execute("""
    INSERT OR IGNORE INTO documents (doc_id, content, source)
    VALUES (?, ?, 'antique')
    """, (row['doc_id'], row['content']))

conn.commit()
print("ANTIQUE dataset successfully imported to SQLite")
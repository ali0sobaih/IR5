import sqlite3
import json

# Connect to SQLite database
conn = sqlite3.connect('ir_project_quora.db')
cursor = conn.cursor()

# Create tables (if they don't exist)
cursor.execute("""
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    title TEXT,
    content TEXT,
    metadata TEXT,
    source TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS queries (
    query_id TEXT PRIMARY KEY,
    query_text TEXT,
    metadata TEXT,
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

# Function to process JSONL files
def process_jsonl(file_path, table_name, source_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if table_name == 'documents':
                cursor.execute("""
                INSERT OR IGNORE INTO documents (doc_id, title, content, metadata, source)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    data['_id'],
                    data.get('title', ''),
                    data.get('text', ''),
                    json.dumps(data.get('metadata', {})),
                    source_name
                ))
            elif table_name == 'queries':
                cursor.execute("""
                INSERT OR IGNORE INTO queries (query_id, query_text, metadata, source)
                VALUES (?, ?, ?, ?)
                """), (
                    data['_id'],
                    data.get('text', ''),
                    json.dumps(data.get('metadata', {})),
                    source_name
                )

# Function to process TSV files
def process_tsv(file_path, source_name):
    with open(file_path, 'r', encoding='utf-8') as f:
        # Skip header if exists
        first_line = f.readline()
        if not first_line.startswith('query-id'):
            f.seek(0)  # Rewind if no header
            
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 3:
                query_id, doc_id, score = parts[0], parts[1], parts[2]
                cursor.execute("""
                INSERT OR IGNORE INTO qrels (query_id, doc_id, relevance, source)
                VALUES (?, ?, ?, ?)
                """, (query_id, doc_id, int(score), source_name))

# Process corpus
process_jsonl('corpus.jsonl', 'documents', 'json_corpus')

# Process queries
process_jsonl('queries.jsonl', 'queries', 'json_queries')

# Process qrels files
process_tsv('qrels/dex.tsv', 'dex')
process_tsv('qrels/test.tsv', 'test')

# Commit changes and close connection
conn.commit()
conn.close()

print("Data successfully imported to SQLite database")
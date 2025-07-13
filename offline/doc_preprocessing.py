import string
import pandas as pd
STOP_WORDS = {
    'a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an',
    'and', 'any', 'are', 'aren', "aren't", 'as', 'at', 'be', 'because', 'been',
    'before', 'being', 'below', 'between', 'both', 'but', 'by', 'can', 'couldn',
    "couldn't", 'd', 'did', 'didn', "didn't", 'do', 'does', 'doesn', "doesn't",
    'doing', 'don', "don't", 'down', 'during', 'each', 'few', 'for', 'from',
    'further', 'had', 'hadn', "hadn't", 'has', 'hasn', "hasn't", 'have', 'haven',
    "haven't", 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself',
    'his', 'how', 'i', 'if', 'in', 'into', 'is', 'isn', "isn't", 'it', "it's",
    'its', 'itself', 'just', 'll', 'm', 'ma', 'me', 'mightn', "mightn't", 'more',
    'most', 'mustn', "mustn't", 'my', 'myself', 'needn', "needn't", 'no', 'nor',
    'not', 'now', 'o', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 're', 's', 'same', 'shan', "shan't",
    'she', "she's", 'should', "should've", 'shouldn', "shouldn't", 'so', 'some',
    'such', 't', 'than', 'that', "that'll", 'the', 'their', 'theirs', 'them',
    'themselves', 'then', 'there', 'these', 'they', 'this', 'those', 'through',
    'to', 'too', 'under', 'until', 'up', 've', 'very', 'was', 'wasn', "wasn't",
    'we', 'were', 'weren', "weren't", 'what', 'when', 'where', 'which', 'while',
    'who', 'whom', 'why', 'will', 'with', 'won', "won't", 'wouldn', "wouldn't",
    'y', 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself',
    'yourselves', 'www', 'com', 'http', 'https', 'could', 'would', 'should',
    'might', 'must', 'may', 'shall', 'also', 'however', 'therefore', 'besides',
    'furthermore', 'meanwhile', 'nevertheless', 'otherwise', 'instead', 'anyway',
    'inc', 'ltd', 'etc', 'eg', 'ie', 'namely', 'including', 'according', 'often',
    'usually', 'sometimes', 'never', 'always', 'among', 'another', 'anything',
    'anywhere', 'every', 'everyone', 'everything', 'everywhere', 'neither',
    'none', 'nothing', 'nowhere', 'several', 'somebody', 'someone', 'something',
    'somewhere', 'whatever', 'whenever', 'wherever', 'whichever', 'whoever',
    'whomever', 'within', 'without', 'upon', 'whether', 'since', 'although',
    'because', 'unless', 'until', 'while', 'regarding', 'following', 'both',
    'either', 'neither', 'each', 'every', 'many', 'most', 'some', 'such', 'even',
    'just', 'like', 'particularly', 'specifically', 'especially', 'namely'
}

def preprocess(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = text.split()  # Simple split (no NLTK needed)
    filtered_tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(filtered_tokens)

# Document Loading antique - SQLite Version 
def load_documents_from_antique_db(db_path='ir_project.db', source='antique'):
    """Load documents from SQLite database with NULL handling"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    # Only select documents with non-NULL content
    collection_df = pd.read_sql(
        "SELECT doc_id as docid, content as text FROM documents WHERE source = ? AND content IS NOT NULL", 
        conn, 
        params=(source,)
    )
    
    # Create dictionaries
    documents = dict(zip(collection_df['docid'], collection_df['text']))
    raw_texts = documents.copy()
    
    conn.close()
    return collection_df, documents, raw_texts

# Load documents
collection_df, documents, raw_texts = load_documents_from_antique_db()

# Preprocess documents
tokenized_docs = []
for docid, text in documents.items():
    processed = preprocess(text)
    documents[docid] = processed
    tokenized_docs.append(processed.split())


    # Document Loading - SQLite Version
def load_documents_from_db(db_path='ir_project.db', source='quora'):
    """Load documents from SQLite database with NULL handling"""
    import sqlite3
    conn = sqlite3.connect(db_path)
    
    # Only select documents with non-NULL content
    collection_df = pd.read_sql(
        "SELECT doc_id as docid, content as text FROM documents WHERE source = ? AND content IS NOT NULL", 
        conn, 
        params=(source,)
    )
    
    # Create dictionaries
    documents = dict(zip(collection_df['docid'], collection_df['text']))
    raw_texts = documents.copy()
    
    conn.close()
    return collection_df, documents, raw_texts

# Load documents
collection_df, documents, raw_texts = load_documents_from_db()

# Preprocess documents
tokenized_docs = []
for docid, text in documents.items():
    processed = preprocess(text)
    documents[docid] = processed
    tokenized_docs.append(processed.split())
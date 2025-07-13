import joblib
import chromadb
from config import CHROMA_PATH, JOBLIB_PATH
from sentence_transformers import SentenceTransformer

# Load existing data
data = joblib.load(JOBLIB_PATH)

# Initialize ChromaDB
client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = client.create_collection("bert_embeddings")

# Store BERT embeddings
collection.add(
    ids=[str(doc_id) for doc_id in data['documents'].keys()],
    embeddings=data['bert_embeddings'].tolist(),
    documents=list(data['raw_texts'].values()))
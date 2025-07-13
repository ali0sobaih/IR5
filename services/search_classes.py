# services/search_classes.py

from services.preprocessing_service import preprocess
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import functools
from services.database_utils import get_doc_text_by_id 

# --- Global Caches for BERT Components ---
@functools.lru_cache(maxsize=None)
def _get_bert_model(model_name='all-MiniLM-L6-v2'):
    print(f"Loading BERT model: {model_name}...")
    return SentenceTransformer(model_name)

class TfIdfSearch:
    def __init__(self, data):
        self.vec = data["vectorizer"]
        self.mat = data["matrix"]
        self.doc_ids = data["doc_ids"]
        self.dataset = data["dataset"]

    def execute_search(self, query):
        vec = self.vec.transform([preprocess(query)])
        scores = (self.mat @ vec.T).toarray().flatten()
        
        # Sort by score and get all indices for scoring
        # Limit to top 10 for efficiency if full list isn't needed by hybrid
        top_idx = scores.argsort()[::-1][:10] # Fetch top 10 documents

        results = []
        for i in top_idx:
            doc_id = self.doc_ids[i]
            doc_text = get_doc_text_by_id(self.dataset, doc_id) 
            results.append({
                "doc_id": doc_id,
                "doc_text": doc_text,
                "score": float(scores[i]) 
            })
        return results

class Bm25Search:
    def __init__(self, data):
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.tokenized = data["tokenized_docs"]
        self.dataset = data["dataset"]

    def execute_search(self, query):
        tokens = preprocess(query).split()
        scores = self.bm25.get_scores(tokens)
        
        # Limit to top 10 documents
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10] 

        results = []
        for i in top_idx:
            doc_id = self.doc_ids[i]
            doc_text = get_doc_text_by_id(self.dataset, doc_id) 
            results.append({
                "doc_id": doc_id,
                "doc_text": doc_text,
                "score": float(scores[i]) 
            })
        return results

class BertSearch:
    def __init__(self, data):
        self.model = _get_bert_model()
        self.dataset = data["dataset"]
        self.doc_ids = data["doc_ids"]

        index_path = f"faiss_store/{self.dataset}/index.faiss"
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index for dataset '{self.dataset}' not found at '{index_path}'. "
                                    "Please ensure it has been pre-computed and saved.")
        try:
            self.index = faiss.read_index(index_path)
            if self.index.ntotal > 0:
                print(f"FAISS index loaded for dataset: {self.dataset} (size: {self.index.ntotal})")
            else:
                print(f"Warning: FAISS index for {self.dataset} is empty. No documents to search.")
        except Exception as e:
            raise RuntimeError(f"Failed to load FAISS index from '{index_path}': {e}")
        
    def execute_search(self, query):
        if not self.index or self.index.ntotal == 0 or not self.doc_ids:
            print(f"Skipping search: Index or document data is empty for this BertSearch instance.")
            return []

        processed = preprocess(query) 
        q_emb = self.model.encode([processed]).astype("float32")
        
        # Search for top 50 as before, then return top 10 with text
        distances, indices = self.index.search(q_emb, 50) 
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.doc_ids): 
                doc_id = self.doc_ids[i]
                doc_text = get_doc_text_by_id(self.dataset, doc_id) 
                bert_score = 1 - (dist / 2) 
                results.append({
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                    "score": float(bert_score) 
                })
        # Return top 10 after getting all relevant docs
        return sorted(results, key=lambda x: x['score'], reverse=True)[:10]

# HybridSearch class is removed from here. Its logic moves to search_service.py.
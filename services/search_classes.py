# services/search_classes.py
from services.preprocessing_service import preprocess
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import sqlite3
import os
import functools

# --- Global Caches for BERT Components ---
@functools.lru_cache(maxsize=None)
def _get_bert_model(model_name='all-MiniLM-L6-v2'):
    print(f"Loading BERT model: {model_name}...")
    return SentenceTransformer(model_name)

# Helper: No longer strictly needed for BertSearch doc loading, as load_data handles it,
# but can remain for cached model loading. Removed _bert_document_data_cache here
# as docs/doc_ids will be passed consistently via `data`.

class TfIdfSearch:
    def __init__(self, data):
        self.vec = data["vectorizer"]
        self.mat = data["matrix"]
        self.doc_ids = data["doc_ids"]
        self.docs = data["docs"] # NEW: Access full document content

    def execute_search(self, query):
        vec = self.vec.transform([preprocess(query)])
        scores = (self.mat @ vec.T).toarray().flatten()
        top_idx = scores.argsort()[::-1][:10]
        
        results = []
        for i in top_idx:
            doc_id = self.doc_ids[i]
            doc_text = self.docs.get(doc_id, "") # Get actual document text
            results.append({
                "doc_id": doc_id,
                "doc_text": doc_text,
                # Minimal info, no 'title', 'snippet', 'score' properties for now
            })
        return results

class Bm25Search:
    def __init__(self, data):
        self.bm25 = data["bm25"]
        self.doc_ids = data["doc_ids"]
        self.tokenized = data["tokenized_docs"]
        self.docs = data["docs"] # NEW: Access full document content

    def execute_search(self, query):
        tokens = preprocess(query).split()
        scores = self.bm25.get_scores(tokens)
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:10]

        results = []
        for i in top_idx:
            doc_id = self.doc_ids[i]
            doc_text = self.docs.get(doc_id, "") # Get actual document text
            results.append({
                "doc_id": doc_id,
                "doc_text": doc_text,
                # Minimal info
            })
        return results

class BertSearch:
    def __init__(self, data): # `data` now consistently contains `docs`, `doc_ids`, `dataset`
        self.model = _get_bert_model()
        self.dataset = data["dataset"]
        self.docs = data["docs"] # Docs provided by load_data
        self.doc_ids = data["doc_ids"] # Doc IDs provided by load_data

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
        
        distances, indices = self.index.search(q_emb, 10) 
        
        results = []
        for i, dist in zip(indices[0], distances[0]):
            if i < len(self.doc_ids):
                doc_id = self.doc_ids[i]
                doc_text = self.docs.get(doc_id, "")
                results.append({
                    "doc_id": doc_id,
                    "doc_text": doc_text,
                    # Minimal info, no 'title', 'snippet', 'score' properties for now
                })
        return results

class HybridSearch:
    # No changes needed here, as it calls the other search services which now return structured data
    def __init__(self, data): # Changed `dataset` to `data` to receive the full data dict
        from services.search_factory import get_search_service

        # Pass the full data object to BERT and TFIDF to ensure they get all necessary info
        self.bert = get_search_service("bert", data) 
        self.tfidf = get_search_service("tfidf", data)

    def execute_search(self, query):
        bert_results = self.bert.execute_search(query)
        tfidf_results = self.tfidf.execute_search(query)
        
        # Combine results; prioritize BERT results, then TFIDF.
        # Ensure uniqueness by doc_id to prevent duplicates
        combined_results_map = {res['doc_id']: res for res in bert_results}
        for res in tfidf_results:
            if res['doc_id'] not in combined_results_map:
                combined_results_map[res['doc_id']] = res

        # Convert back to list and take top 10
        return list(combined_results_map.values())[:10]
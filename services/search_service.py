# services/search_service.py

from fastapi import APIRouter, Body, HTTPException
from pydantic import BaseModel, Field
import joblib
import sqlite3
import functools
from services.query_expansion_service import expand_query_with_synonyms
from services.database_utils import get_doc_text_by_id # Still used by the search classes
from services.search_classes import TfIdfSearch, Bm25Search, BertSearch
import httpx # For making HTTP requests to other local endpoints
import asyncio # <--- ADDED THIS IMPORT

router = APIRouter()

# --- Pydantic Models ---
class RefineQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="The original query string to expand.")

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="The query string to search with.")
    dataset: str = Field(..., description="The dataset to search in (e.g., 'antique', 'quora')")

class SearchResult(BaseModel):
    doc_id: str
    doc_text: str
    score: float

# --- Cached Data Loading ---
@functools.lru_cache(maxsize=None)
def _cached_load_data(search_type: str, dataset: str):
    """
    Loads and caches the necessary data (vectorizer, matrix, bm25, doc_ids, faiss index)
    for a given search_type and dataset.
    This now specifically loads only the components relevant to the requested search_type.
    """
    joblib_data = {}
    
    # Load search-type specific joblib data
    joblib_path = f"offline_data/{search_type}_{dataset}.joblib"
    try:
        loaded_joblib = joblib.load(joblib_path)
        joblib_data.update(loaded_joblib)
        print(f"Loaded {search_type} model data for dataset '{dataset}' from '{joblib_path}'.")
    except FileNotFoundError:
        # BERT does not have a joblib model file, its FAISS index is loaded separately in BertSearch
        if search_type not in ["bert", "hybrid"]: # Hybrid doesn't load direct models
            print(f"Warning: Offline data for {search_type} and dataset {dataset} not found at {joblib_path}. "
                  "This might be expected for BERT (index loaded by class) or Hybrid (orchestrates).")
    except Exception as e:
        print(f"Error loading joblib data for {search_type} from {joblib_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to load search model for {search_type}: {e}")


    # Always load doc_ids list from SQLite. This list is needed for mapping.
    conn = None
    doc_ids_list = []
    try:
        db_path = "offline/ir_project.db" # Make sure this path is correct relative to your app's root
        conn = sqlite3.connect(db_path)
        cursor = conn.execute(f"SELECT doc_id FROM `{dataset}` ORDER BY doc_id ASC") 
        doc_ids_list = [row[0] for row in cursor.fetchall()]
        print(f"Loaded {len(doc_ids_list)} document IDs for dataset '{dataset}' from '{db_path}'.")
    except sqlite3.OperationalError as e:
        print(f"SQLite error loading document IDs for '{dataset}': {e}. Ensure table `{dataset}` exists in '{db_path}' and is accessible.")
        raise HTTPException(status_code=500, detail=f"Failed to load document IDs for dataset {dataset}: {e}")
    finally:
        if conn:
            conn.close()
    
    joblib_data["dataset"] = dataset 
    joblib_data["doc_ids"] = doc_ids_list 

    return joblib_data

# --- Search Service Instances Cache (per dataset/type) ---
# This dictionary will hold the instantiated search class objects (TfIdfSearch, Bm25Search, BertSearch)
# preventing re-initialization on every request for the same dataset/type.
_search_service_instances = {}


# --- API Endpoints ---

@router.post("/refineQuery/")
async def refine_query(
    request: RefineQueryRequest 
):
    """
    Expands the given query using synonym expansion and returns the original and expanded query.
    """
    expanded_query = expand_query_with_synonyms(request.query) 
    print("original_query: "+ request.query) 
    print("expanded_query: "+ expanded_query)
    return {"original_query": request.query, "expanded_query": expanded_query}


@router.post("/search/tfidf", response_model=list[SearchResult])
async def search_tfidf(req: SearchRequest):
    """Performs TFIDF search for the given query and dataset."""
    try:
        data_payload = _cached_load_data("tfidf", req.dataset)
        cache_key = ("tfidf", req.dataset)
        
        if cache_key not in _search_service_instances:
            print(f"Initializing TfIdfSearch for {req.dataset}...")
            service = TfIdfSearch(data_payload)
            _search_service_instances[cache_key] = service
        else:
            service = _search_service_instances[cache_key]
        
        results = service.execute_search(req.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TFIDF Search Error: {e}")

@router.post("/search/bm25", response_model=list[SearchResult])
async def search_bm25(req: SearchRequest):
    """Performs BM25 search for the given query and dataset."""
    try:
        data_payload = _cached_load_data("bm25", req.dataset)
        cache_key = ("bm25", req.dataset)
        
        if cache_key not in _search_service_instances:
            print(f"Initializing Bm25Search for {req.dataset}...")
            service = Bm25Search(data_payload)
            _search_service_instances[cache_key] = service
        else:
            service = _search_service_instances[cache_key]
        
        results = service.execute_search(req.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BM25 Search Error: {e}")

@router.post("/search/bert", response_model=list[SearchResult])
async def search_bert(req: SearchRequest):
    """Performs BERT search for the given query and dataset."""
    try:
        data_payload = _cached_load_data("bert", req.dataset)
        cache_key = ("bert", req.dataset)
        
        if cache_key not in _search_service_instances:
            print(f"Initializing BertSearch for {req.dataset}...")
            service = BertSearch(data_payload)
            _search_service_instances[cache_key] = service
        else:
            service = _search_service_instances[cache_key]
        
        results = service.execute_search(req.query)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"BERT Search Error: {e}")


@router.post("/search/hybrid", response_model=list[SearchResult])
async def search_hybrid(req: SearchRequest):
    """
    Performs a hybrid search by combining results from BERT, TFIDF, and BM25 APIs.
    """
    # Define weights
    weights = {
        "bert": 0.5,
        "tfidf": 0.2,
        "bm25": 0.3
    }
    
    # asynchronous HTTP requests to local endpoints
    async with httpx.AsyncClient() as client:
        bert_task = client.post(
            "http://127.0.0.1:8000/api/search/bert", 
            json={"query": req.query, "dataset": req.dataset}
        )
        tfidf_task = client.post(
            "http://127.0.0.1:8000/api/search/tfidf",
            json={"query": req.query, "dataset": req.dataset}
        )
        bm25_task = client.post(
            "http://127.00.1:8000/api/search/bm25",
            json={"query": req.query, "dataset": req.dataset}
        )

        bert_response, tfidf_response, bm25_response = await asyncio.gather(
            bert_task, tfidf_task, bm25_task, return_exceptions=True 
        )

    bert_results = []
    if isinstance(bert_response, httpx.Response) and bert_response.status_code == 200:
        bert_results = bert_response.json()
    else:
        print(f"Warning: BERT search failed or returned non-200: {bert_response}")

    tfidf_results = []
    if isinstance(tfidf_response, httpx.Response) and tfidf_response.status_code == 200:
        tfidf_results = tfidf_response.json()
    else:
        print(f"Warning: TFIDF search failed or returned non-200: {tfidf_response}")

    bm25_results = []
    if isinstance(bm25_response, httpx.Response) and bm25_response.status_code == 200:
        bm25_results = bm25_response.json()
    else:
        print(f"Warning: BM25 search failed or returned non-200: {bm25_response}")

    # Normalize scores
    def normalize_scores(results_list):
        if not results_list:
            return results_list
        scores = [res["score"] for res in results_list if "score" in res]
        if not scores:
            return results_list
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score: 
            for res in results_list:
                if "score" in res:
                    res["normalized_score"] = 0.0 
            else:
                for res in results_list: # This 'else' block will never be reached, remove for clarity
                    if "score" in res:
                        res["normalized_score"] = 0.0 # Assign 0 if min and max are the same
        else:
            for res in results_list:
                if "score" in res:
                    res["normalized_score"] = (res["score"] - min_score) / (max_score - min_score)
        return results_list

    bert_results = normalize_scores(bert_results)
    tfidf_results = normalize_scores(tfidf_results)
    bm25_results = normalize_scores(bm25_results)

    # Combine scores
    combined_scores_map = {} 

    for res in bert_results:
        doc_id = res["doc_id"]
        # Ensure doc_text is carried over as it's needed for final results
        combined_scores_map.setdefault(doc_id, {"score": 0.0, "doc_text": res.get("doc_text", "")})
        combined_scores_map[doc_id]["score"] += res.get("normalized_score", 0.0) * weights["bert"]

    for res in tfidf_results:
        doc_id = res["doc_id"]
        combined_scores_map.setdefault(doc_id, {"score": 0.0, "doc_text": res.get("doc_text", "")})
        combined_scores_map[doc_id]["score"] += res.get("normalized_score", 0.0) * weights["tfidf"]
        # If doc_text was not present from BERT, take it from TFIDF (or vice-versa)
        if not combined_scores_map[doc_id]["doc_text"] and res.get("doc_text"):
            combined_scores_map[doc_id]["doc_text"] = res["doc_text"]
        
    for res in bm25_results:
        doc_id = res["doc_id"]
        combined_scores_map.setdefault(doc_id, {"score": 0.0, "doc_text": res.get("doc_text", "")})
        combined_scores_map[doc_id]["score"] += res.get("normalized_score", 0.0) * weights["bm25"]
        # If doc_text was not present from other sources, take it from BM25
        if not combined_scores_map[doc_id]["doc_text"] and res.get("doc_text"):
            combined_scores_map[doc_id]["doc_text"] = res["doc_text"]

    # Sort final results
    final_results = []
    for doc_id, data in combined_scores_map.items():
        final_results.append({
            "doc_id": doc_id,
            "doc_text": data["doc_text"],
            "score": data["score"]
        })

    # Sort by score in descending order and return top 10
    final_results.sort(key=lambda x: x["score"], reverse=True)
    
    return final_results[:10]
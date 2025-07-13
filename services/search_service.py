from fastapi import APIRouter, Body
from pydantic import BaseModel, Field
import joblib
import sqlite3 # Import sqlite3 for document loading
from services.search_factory import get_search_service
from services.query_expansion_service import expand_query_with_synonyms

router = APIRouter()

class RefineQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=100, description="The original query string to expand.")


def load_data(search_type, dataset):
    joblib_data = {}
    
    # Attempt to load search-type specific joblib data if it exists.
    # This keeps the existing loading logic for TFIDF/BM25.
    # For BERT, this joblib file might not exist, which is fine as its core data is elsewhere.
    try:
        joblib_path = f"offline_data/{search_type}_{dataset}.joblib"
        loaded_joblib = joblib.load(joblib_path)
        joblib_data.update(loaded_joblib) # Merge loaded data
    except FileNotFoundError:
        # Only raise if it's a non-BERT/Hybrid search type that *should* have a joblib file
        if search_type not in ["bert", "hybrid"]:
            raise FileNotFoundError(f"Offline data for {search_type} and dataset {dataset} not found at {joblib_path}. "
                                    "Please ensure it has been pre-computed.")

    # Always load actual document content and doc_ids from SQLite for all search types
    # This ensures all search classes have access to document text for display.
    conn = None
    docs_dict = {}
    doc_ids_list = []
    try:
        conn = sqlite3.connect("offline/ir_project.db")
        cursor = conn.execute(f"SELECT doc_id, doc FROM `{dataset}`")
        for row in cursor:
            doc_id, doc_content = row
            docs_dict[doc_id] = doc_content
            doc_ids_list.append(doc_id)
    except sqlite3.OperationalError as e:
        print(f"SQLite error loading documents for '{dataset}' in load_data: {e}. "
              f"Ensure table `{dataset}` exists in 'offline/ir_project.db' and is accessible.")
        # Continue with empty docs_dict if error occurs, to avoid crashing
    finally:
        if conn:
            conn.close()
    
    joblib_data["dataset"] = dataset # Ensure dataset name is passed
    joblib_data["docs"] = docs_dict # Add the loaded document content
    joblib_data["doc_ids"] = doc_ids_list # Add the list of doc IDs

    return joblib_data

# API Endpoint for Query Refinement
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


@router.post("/searchScreen/")
async def search(
    query: str = Body(..., min_length=1, max_length=100, description="The *already expanded* query string to search with."),
    dataset: str = Body(..., description="The dataset to search in (e.g., 'antique', 'quora')"),
    search_type: str = Body(..., description="The type of search to perform (e.g., 'tfidf', 'bm25', 'bert', 'hybrid')")
):
    """
    Performs a search using the provided (and expected to be pre-expanded) query, dataset, and search type.
    """
    # The query is now expected to be already expanded by the frontend
    # Removed the direct call to expand_query_with_synonyms here.
    # print(f"Received query for search: '{query}'") # You can uncomment this for debugging if needed

    data = load_data("bert" if search_type == "bert" else search_type, dataset)
    service = get_search_service(search_type, data)

    # Use the 'query' parameter directly, assuming it's already expanded
    return service.execute_search(query)
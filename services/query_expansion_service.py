# services/query_expansion_service.py (Updated with Semantic Logic Fixes)

import string
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import os
from pathlib import Path

# --- Global Model Initialization ---
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Warning: Could not load SentenceTransformer model. Ensure 'all-MiniLM-L6-v2' is available. Error: {e}")
    model = None

# --- Global Vocabulary and FAISS Index Loading - HARDCODED PATHS ---
# Directory where the FAISS index and vocabulary are located
# This MUST match the FAISS_STORE path used in dataPipline.py
FAISS_STORE = Path("C:/Users/Lenovo/Desktop/5th/second simester/IR/IR_PROJECT/offline_data")

VOCAB_FILE_PATH = FAISS_STORE / "general_semantic_vocabulary.joblib"
FAISS_INDEX_PATH = FAISS_STORE / "general_semantic_vocabulary.faiss"

vocabulary_words = None
faiss_index = None

try:
    if VOCAB_FILE_PATH.exists() and FAISS_INDEX_PATH.exists():
        vocabulary_words = joblib.load(VOCAB_FILE_PATH)
        faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
        print(f"Loaded semantic vocabulary with {len(vocabulary_words)} words and FAISS index.")
    else:
        print(f"Warning: Semantic vocabulary files not found at {FAISS_STORE}. Query expansion will be limited.")
except Exception as e:
    print(f"Error loading semantic vocabulary or FAISS index: {e}")
    vocabulary_words = None
    faiss_index = None

# --- Adjusted get_semantic_synonyms function ---
def get_semantic_synonyms(word, top_n=3, cosine_similarity_threshold=0.7):
    """
    Finds semantically similar words using SentenceTransformer embeddings
    and a pre-built FAISS index, filtering by cosine similarity.
    """
    if not model or vocabulary_words is None or faiss_index is None:
        return []

    # Encode the query word. SentenceTransformer's encode by default produces normalized embeddings.
    query_embedding = model.encode([word], convert_to_tensor=False).astype('float32')

    # Perform similarity search. Search for more candidates than needed for filtering.
    # L2 distance is used with IndexFlatL2
    # Search for more candidates (e.g., 10 or top_n * 5) to ensure we can find 'top_n' after filtering
    distances, indices = faiss_index.search(query_embedding, max(10, top_n * 5))

    synonyms = []
    for i, idx in enumerate(indices[0]):
        if idx == -1: # No valid result at this index
            continue

        # Convert L2 distance to cosine similarity (assuming normalized embeddings)
        # Formula: cosine_similarity = 1 - (L2_distance^2 / 2)
        l2_distance_squared = distances[0][i]
        current_cosine_similarity = 1 - (l2_distance_squared / 2)

        candidate_word = vocabulary_words[idx]

        # Filter by cosine similarity threshold, exclude the original word, and limit to top_n
        if current_cosine_similarity >= cosine_similarity_threshold and \
           candidate_word.lower() != word.lower() and \
           len(synonyms) < top_n:
            synonyms.append(candidate_word)

    return synonyms

def expand_query_with_synonyms(query, top_n_synonyms_per_word=1): # Default top_n is now 1
    """
    Expands a query by adding semantically similar terms for each word
    using SentenceTransformer embeddings and a FAISS index.
    The original query words are always included.
    """
    expanded_terms = []
    cleaned_query = query.lower().translate(str.maketrans('', '', string.punctuation))
    words = cleaned_query.split()

    for word in words:
        expanded_terms.append(word)
        # Call get_semantic_synonyms with your desired top_n and cosine_similarity_threshold
        # For testing, you might want to try top_n_synonyms_per_word=3 or 5 to see more results
        semantic_syns = get_semantic_synonyms(word, top_n=top_n_synonyms_per_word, cosine_similarity_threshold=0.7)
        for syn in semantic_syns:
            expanded_terms.append(syn)

    return " ".join(list(dict.fromkeys(expanded_terms)))
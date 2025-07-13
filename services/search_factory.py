from services.search_classes import TfIdfSearch, Bm25Search, BertSearch, HybridSearch

def get_search_service(search_type, dataset):
    if search_type == "tfidf":
        return TfIdfSearch(dataset)
    elif search_type == "bm25":
        return Bm25Search(dataset)
    elif search_type == "bert":
        return BertSearch(dataset)
    elif search_type == "hybrid":
        return HybridSearch(dataset)
    else:
        raise ValueError("Invalid search type")

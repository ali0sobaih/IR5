import json
import numpy as np
import os
import faiss
from pathlib import Path
from sentence_transformers import SentenceTransformer
from langchain.schema import BaseRetriever, Document
from pydantic import Field, PrivateAttr # Import PrivateAttr

class CustomFaissRetriever(BaseRetriever):
    # Declare these as Pydantic fields. They will be passed to the constructor
    # and Pydantic will manage their assignment.
    embeddings_model_name: str
    faiss_index_paths: dict
    data_file_paths: dict

    # Use PrivateAttr for internal objects (SentenceTransformer, FAISS indexes, lists of text)
    # Pydantic will not try to validate or serialize these, but they are still instance attributes.
    _embeddings_model: SentenceTransformer = PrivateAttr()
    _faiss_index_antique: faiss.IndexFlatL2 = PrivateAttr()
    _faiss_index_quora: faiss.IndexFlatL2 = PrivateAttr()
    _antique_docs_text: list[str] = PrivateAttr()
    _quora_docs_text: list[str] = PrivateAttr()

    # The constructor now takes all arguments via **data and passes them to super().__init__
    # Pydantic automatically maps the arguments to the declared fields.
    def __init__(self, **data):
        super().__init__(**data)

        # Now, access the Pydantic-managed fields using self.<field_name>
        # and use them to initialize your internal PrivateAttr attributes.
        self._embeddings_model = SentenceTransformer(self.embeddings_model_name)
        print(f"CustomFaissRetriever: Initializing with embedding model '{self.embeddings_model_name}'")

        # Load raw FAISS indexes
        if not all(Path(p).exists() for p in self.faiss_index_paths.values()):
            raise FileNotFoundError(f"One or more FAISS index files not found: {self.faiss_index_paths.values()}")
        self._faiss_index_antique = faiss.read_index(str(self.faiss_index_paths["antique"]))
        self._faiss_index_quora = faiss.read_index(str(self.faiss_index_paths["quora"]))
        print("CustomFaissRetriever: Loaded raw FAISS indexes.")

        # Load document texts from files
        if not all(Path(p).exists() for p in self.data_file_paths.values()):
            raise FileNotFoundError(f"One or more data files not found: {self.data_file_paths.values()}")
        self._antique_docs_text = self._load_antique_text(self.data_file_paths["antique"])
        self._quora_docs_text = self._load_quora_text(self.data_file_paths["quora"])
        print("CustomFaissRetriever: Loaded document texts from files.")


    def _load_antique_text(self, file_path: Path) -> list[str]:
        """Loads text from antique/collection.txt (one doc per line)."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_quora_text(self, file_path: Path) -> list[str]:
        """Loads text from quora/corpus.jsonl (JSON objects, 'text' field)."""
        docs = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if 'text' in data:
                    docs.append(data['text'])
        return docs

    def get_relevant_documents(self, query: str) -> list[Document]:
        # Embed the query using the internal _embeddings_model
        query_embedding = self._embeddings_model.encode([query], convert_to_tensor=False).astype("float32")

        k_per_source = 2
        
        retrieved_docs_with_distances = []

        # Search antique index
        D_antique, I_antique = self._faiss_index_antique.search(query_embedding, k_per_source)
        for i, idx in enumerate(I_antique[0]):
            if idx != -1 and 0 <= idx < len(self._antique_docs_text):
                doc_content = self._antique_docs_text[idx]
                distance = D_antique[0][i]
                retrieved_docs_with_distances.append(
                    (distance, Document(page_content=doc_content, metadata={"source": "antique", "faiss_id": idx}))
                )
            else:
                print(f"Warning: Antique index {idx} out of bounds or -1.")

        # Search quora index
        D_quora, I_quora = self._faiss_index_quora.search(query_embedding, k_per_source)
        for i, idx in enumerate(I_quora[0]):
            if idx != -1 and 0 <= idx < len(self._quora_docs_text):
                doc_content = self._quora_docs_text[idx]
                distance = D_quora[0][i]
                retrieved_docs_with_distances.append(
                    (distance, Document(page_content=doc_content, metadata={"source": "quora", "faiss_id": idx}))
                )
            else:
                print(f"Warning: Quora index {idx} out of bounds or -1.")

        retrieved_docs_with_distances.sort(key=lambda x: x[0])
        return [doc for dist, doc in retrieved_docs_with_distances]

    async def aget_relevant_documents(self, query: str) -> list[Document]:
        return await super().aget_relevant_documents(query)
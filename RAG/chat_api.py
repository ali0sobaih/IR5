# RAG/chat_api.py

from fastapi import APIRouter, Query as FastQuery
from pydantic import BaseModel
from langchain.chains import RetrievalQA
from langchain_cohere import ChatCohere
from langchain.prompts import PromptTemplate
from .redundant_filter_retriever import CustomFaissRetriever
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

router = APIRouter()

# --- Define paths ---
BASE_DIR = Path(__file__).parent.parent
FAISS_STORE_PATH = BASE_DIR / "faiss_store"
DATA_PATH = BASE_DIR / "data"

ANTIQUE_FAISS_INDEX = FAISS_STORE_PATH / "antique" / "index.faiss"
QUORA_FAISS_INDEX = FAISS_STORE_PATH / "quora" / "index.faiss"

ANTIQUE_DATA_FILE = DATA_PATH / "antique" / "collection.txt"
QUORA_DATA_FILE = DATA_PATH / "quora" / "corpus.jsonl"

# --- Initialize Cohere Chat Model ---
chat = ChatCohere(model="command-r", verbose=True)

# --- Initialize the CustomFaissRetriever ---
try:
    retriever = CustomFaissRetriever(
        embeddings_model_name="all-MiniLM-L6-v2",
        faiss_index_paths={
            "antique": ANTIQUE_FAISS_INDEX,
            "quora": QUORA_FAISS_INDEX
        },
        data_file_paths={
            "antique": ANTIQUE_DATA_FILE,
            "quora": QUORA_DATA_FILE
        }
    )
    print("Successfully initialized CustomFaissRetriever.")
except Exception as e:
    print(f"Error initializing CustomFaissRetriever: {e}")
    print("Please ensure process_bert.py has been run and the data/FAISS files are correctly located.")
    raise e

prompt_template = PromptTemplate.from_template(
    "Answer briefly and concisely. Be direct and use only the relevant context below:\n\n{context}\n\nQuestion: {question}\nAnswer:"
)

qa_chain = RetrievalQA.from_chain_type(
    llm=chat,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt_template},
    return_source_documents=True
)

class ChatQuery(BaseModel):
    question: str

app = APIRouter()

@router.post("/chat/")
async def rag_chat(query: ChatQuery):
    response = qa_chain.invoke(query.question)
    
    answer = response.get("result")
    source_documents = response.get("source_documents", [])

    formatted_sources = []
    for doc in source_documents:
        # Create a mutable copy of metadata to modify it
        metadata_copy = doc.metadata.copy()
        
        # Explicitly convert 'faiss_id' to a standard Python int if it exists
        if "faiss_id" in metadata_copy:
            metadata_copy["faiss_id"] = int(metadata_copy["faiss_id"]) 
            
        formatted_sources.append({
            "page_content": doc.page_content,
            "metadata": metadata_copy # Use the modified copy
        })

    return {
        "answer": answer,
        "sources": formatted_sources
    }
from fastapi import FastAPI
from services.search_service import router as search_router
from starlette.middleware.cors import CORSMiddleware 
from RAG.chat_api import router as chat_router

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000", 
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # Allows specific origins
    allow_credentials=True,         # Allow cookies to be included in requests
    allow_methods=["*"],            # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],            # Allows all headers
)

app.include_router(search_router, prefix="/api")
app.include_router(chat_router, prefix="/api")


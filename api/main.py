# api/main.py
# ============================================================
# FastAPI application factory — Phase 3
#
# Run:
#   uvicorn api.main:app --reload --port 8000
#
# Docs auto-generated at:
#   http://localhost:8000/docs      (Swagger UI)
#   http://localhost:8000/redoc     (ReDoc)
# ============================================================

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import chat, documents, health
from config.settings import get_settings
from core.ingestion.embedder import get_embedding_model
from core.rag.reranker import CrossEncoderReranker
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: pre-load heavy models so the first request isn't slow.
    Shutdown: nothing to clean up (models unload with the process).
    """
    logger.info("🚀 Starting RAG Assistant API...")
    logger.info(f"   LLM:        {settings.groq_model} via Groq")
    logger.info(f"   Embeddings: {settings.embedding_model}")
    logger.info(f"   Reranker:   {settings.reranker_model}")

    # Pre-warm embedding model (downloads on first run, ~100-340MB)
    logger.info("⏳ Pre-loading embedding model...")
    get_embedding_model()

    # Pre-warm reranker
    logger.info("⏳ Pre-loading reranker...")
    CrossEncoderReranker()

    logger.info("✅ API ready.")
    yield
    logger.info("👋 Shutting down.")


app = FastAPI(
    title="RAG Assistant API",
    description=(
        "Document Q&A system. Upload PDFs, ask questions, get cited answers.\n\n"
        "**Stack:** Groq LLaMA 3.3 70B · BAAI/bge-base-en-v1.5 · ChromaDB · Redis"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# ── CORS — allow the React frontend to call the API ───────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.get("/", include_in_schema=False)
async def root():
    return {
        "name": "RAG Assistant API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
    }
# api/routers/health.py
# ============================================================
# Health check endpoint — useful for monitoring / Docker healthcheck
#   GET /health
# ============================================================

from fastapi import APIRouter
from core.cache.redis_cache import RedisCache
from core.vectorstore.chroma_store import ChromaVectorStore
from config.settings import get_settings
from models.schemas import HealthResponse

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    store = ChromaVectorStore()
    cache = RedisCache()
    return HealthResponse(
        status="ok",
        vector_store_chunks=store.total_documents,
        redis_connected=cache.is_connected,
        groq_model=settings.groq_model,
        embedding_model=settings.embedding_model,
    )
# config/settings.py
# ============================================================
# Central configuration — Phase 1 + Phase 2
# ============================================================

from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    # ── LLM — Groq ───────────────────────────────────────────
    groq_api_key: str = Field(..., env="GROQ_API_KEY")
    groq_model: str = Field("llama-3.3-70b-versatile", env="GROQ_MODEL")

    # ── Embeddings — HuggingFace (local) ─────────────────────
    embedding_provider: str = Field("huggingface", env="EMBEDDING_PROVIDER")
    embedding_model: str = Field("BAAI/bge-base-en-v1.5", env="EMBEDDING_MODEL")
    embedding_device: str = Field("cpu", env="EMBEDDING_DEVICE")

    # ── Vector DB ────────────────────────────────────────────
    chroma_persist_dir: str = Field("./data/chroma_db", env="CHROMA_PERSIST_DIR")
    chroma_collection_name: str = Field("rag_documents", env="CHROMA_COLLECTION_NAME")

    # ── File Upload ──────────────────────────────────────────
    max_pdf_files: int = Field(5, env="MAX_PDF_FILES")
    max_pdf_size_mb: int = Field(20, env="MAX_PDF_SIZE_MB")
    upload_dir: str = Field("./data/uploads", env="UPLOAD_DIR")

    # ── Chunking ─────────────────────────────────────────────
    chunk_size: int = Field(1000, env="CHUNK_SIZE")
    chunk_overlap: int = Field(200, env="CHUNK_OVERLAP")

    # ── Retrieval ────────────────────────────────────────────
    top_k_retrieval: int = Field(10, env="TOP_K_RETRIEVAL")
    top_k_final: int = Field(5, env="TOP_K_FINAL")

    # ── Phase 2: Re-ranking ──────────────────────────────────
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL"
    )
    # Alternatives (all free/local via sentence-transformers):
    #   cross-encoder/ms-marco-MiniLM-L-12-v2   (slower, better quality)
    #   cross-encoder/ms-marco-electra-base      (best quality, most RAM)

    # ── Phase 2: Redis Cache ─────────────────────────────────
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_cache_ttl: int = Field(3600, env="REDIS_CACHE_TTL_SECONDS")
    # TTL in seconds — 3600 = 1 hour. Set to 0 to disable expiry.

    # ── Phase 2: Query Expansion ─────────────────────────────
    query_expansion_enabled: bool = Field(True, env="QUERY_EXPANSION_ENABLED")
    query_expansion_n: int = Field(3, env="QUERY_EXPANSION_N")
    # Number of alternative phrasings to generate per query

    # ── Logging ──────────────────────────────────────────────
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    @property
    def upload_path(self) -> Path:
        p = Path(self.upload_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def chroma_path(self) -> Path:
        p = Path(self.chroma_persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def max_pdf_size_bytes(self) -> int:
        return self.max_pdf_size_mb * 1024 * 1024

    @property
    def llm_model_name(self) -> str:
        return self.groq_model


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
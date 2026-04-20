# models/schemas.py
# ============================================================
# Pydantic data models — Phase 1 + 2 + 3
# ============================================================

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class _Base(BaseModel):
    model_config = {"protected_namespaces": ()}


# ── Ingestion ────────────────────────────────────────────────

class DocumentChunk(_Base):
    chunk_id: str
    source_file: str
    file_hash: str
    page_number: int
    chunk_index: int
    content: str
    token_count: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestionResult(_Base):
    filename: str
    file_hash: str
    total_pages: int
    total_chunks: int
    was_duplicate: bool = False
    error: Optional[str] = None


class BatchIngestionResult(_Base):
    processed: List[IngestionResult]
    total_chunks_added: int
    duration_seconds: float


# ── Retrieval & RAG ──────────────────────────────────────────

class RetrievedChunk(_Base):
    model_config = {"protected_namespaces": (), "coerce_numbers_to_str": True}
    chunk_id: str
    content: str
    source_file: str
    page_number: int
    similarity_score: float
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryRequest(_Base):
    question: str = Field(..., min_length=1, max_length=2000)
    top_k: Optional[int] = None
    filter_filename: Optional[str] = None


class QueryResponse(_Base):
    question: str
    answer: str
    sources: List[RetrievedChunk]
    model_used: str
    retrieval_count: int
    duration_seconds: float
    from_cache: bool = False


# ── Phase 3: API-specific schemas ────────────────────────────

class IndexedFile(_Base):
    """Metadata for a file currently indexed in the vector store."""
    filename: str
    chunk_count: int


class IndexedFilesResponse(_Base):
    files: List[IndexedFile]
    total_files: int
    total_chunks: int


class DeleteFileResponse(_Base):
    filename: str
    chunks_deleted: int
    success: bool


class HealthResponse(_Base):
    status: str
    vector_store_chunks: int
    redis_connected: bool
    groq_model: str
    embedding_model: str


class StreamChunk(_Base):
    """Single SSE payload during streaming response."""
    type: str                        # "token" | "sources" | "done" | "error"
    content: Optional[str] = None   # token text
    sources: Optional[List[RetrievedChunk]] = None
    metadata: Optional[Dict[str, Any]] = None
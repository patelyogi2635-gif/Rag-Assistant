# core/vectorstore/chroma_store.py
# ============================================================
# ChromaDB interface — add, query, delete, check duplicates.
#
# Design:
#   - One collection for all documents (filter by metadata).
#   - Duplicate detection via file_hash — re-uploading same PDF
#     is a no-op (idempotent ingestion).
#   - All CRUD here; RAG chain never touches ChromaDB directly.
# ============================================================

from typing import List, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_chroma import Chroma

from config.settings import get_settings
from core.ingestion.embedder import get_embedding_model
from models.schemas import DocumentChunk, RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class ChromaVectorStore:
    """
    Manages the ChromaDB vector store for RAG retrieval.

    Usage:
        store = ChromaVectorStore()
        store.add_chunks(chunks)
        results = store.similarity_search("What is my deductible?", top_k=10)
    """

    def __init__(self):
        self._embedding_model = get_embedding_model()
        self._store = self._build_store()
        logger.info(
            f"📦 ChromaDB ready | collection='{settings.chroma_collection_name}' "
            f"| docs={self.total_documents}"
        )

    # ── Public API ───────────────────────────────────────────

    def add_chunks(self, chunks: List[DocumentChunk]) -> int:
        """
        Add chunks to the vector store.
        Skips duplicates by chunk_id (ChromaDB upsert semantics).
        Returns the number of chunks actually added.
        """
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        metadatas = [c.metadata for c in chunks]
        ids = [c.chunk_id for c in chunks]

        self._store.add_texts(texts=texts, metadatas=metadatas, ids=ids)
        logger.info(f"  ✅ Added {len(chunks)} chunks to vector store.")
        return len(chunks)

    def is_file_indexed(self, file_hash: str) -> bool:
        """
        Check if a file (by SHA-256 hash) is already in the vector store.
        Prevents duplicate ingestion of the same file.
        """
        results = self._store.get(
            where={"file_hash": {"$eq": file_hash}},
            limit=1,
        )
        return len(results["ids"]) > 0

    def similarity_search(
        self,
        query: str,
        top_k: int = settings.top_k_retrieval,
        filter_filename: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Retrieve top_k most similar chunks to the query.
        Optionally filter to a specific source file.
        """
        where_filter = None
        if filter_filename:
            where_filter = {"source": {"$eq": filter_filename}}

        raw_results = self._store.similarity_search_with_relevance_scores(
            query=query,
            k=top_k,
            filter=where_filter,
        )

        retrieved = []
        for doc, score in raw_results:
            retrieved.append(
                RetrievedChunk(
                    chunk_id=doc.metadata.get("chunk_index", ""),
                    content=doc.page_content,
                    source_file=doc.metadata.get("source", "unknown"),
                    page_number=doc.metadata.get("page", 0),
                    similarity_score=round(score, 4),
                    metadata=doc.metadata,
                )
            )

        logger.info(
            f"🔍 Retrieved {len(retrieved)} chunks for query: "
            f"'{query[:60]}...'"
        )
        return retrieved

    def delete_file(self, filename: str) -> int:
        """Delete all chunks belonging to a specific file."""
        results = self._store.get(where={"source": {"$eq": filename}})
        ids_to_delete = results["ids"]

        if ids_to_delete:
            self._store.delete(ids=ids_to_delete)
            logger.info(f"🗑️  Deleted {len(ids_to_delete)} chunks for '{filename}'.")

        return len(ids_to_delete)

    def list_indexed_files(self) -> List[str]:
        """Return unique filenames currently in the vector store."""
        results = self._store.get()
        filenames = set()
        for meta in results["metadatas"]:
            if meta and "source" in meta:
                filenames.add(meta["source"])
        return sorted(filenames)

    @property
    def total_documents(self) -> int:
        """Total number of chunks in the store."""
        return self._store._collection.count()

    # ── Private ──────────────────────────────────────────────

    def _build_store(self) -> Chroma:
        """Initialize or load existing ChromaDB collection."""
        client = chromadb.PersistentClient(
            path=str(settings.chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        return Chroma(
            client=client,
            collection_name=settings.chroma_collection_name,
            embedding_function=self._embedding_model,
        )
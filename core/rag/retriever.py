# core/rag/retriever.py
# ============================================================
# Retriever — Phase 2: vector search + query expansion + re-ranking.
#
# Full pipeline per query:
#   1. Query expansion  → [q_original, q_alt1, q_alt2, q_alt3]
#   2. Vector search    → retrieve top_k_retrieval per variant
#   3. Deduplicate      → merge by chunk_id (no duplicate chunks)
#   4. Cross-encoder    → re-rank merged set, keep top_k_final
# ============================================================

from typing import List, Optional

from config.settings import get_settings
from core.vectorstore.chroma_store import ChromaVectorStore
from core.rag.reranker import CrossEncoderReranker
from core.rag.query_expander import QueryExpander
from models.schemas import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class Retriever:
    """
    Phase 2 retriever: expansion → multi-query search → dedup → rerank.
    """

    def __init__(self, vector_store: ChromaVectorStore):
        self._store = vector_store
        self._reranker = CrossEncoderReranker()
        self._expander = QueryExpander()

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_filename: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """
        Full Phase 2 retrieval pipeline.
        Returns top_k_final chunks, re-ranked by true relevance.
        """
        # Step 1: Expand query into multiple variants
        query_variants = self._expander.expand(query)

        # Step 2: Retrieve candidates for every variant
        all_chunks = self._retrieve_multi(
            queries=query_variants,
            top_k=top_k or settings.top_k_retrieval,
            filter_filename=filter_filename,
        )

        # Step 3: Deduplicate by chunk_id (keep best score per chunk)
        unique_chunks = self._deduplicate(all_chunks)
        logger.info(
            f"  → {len(all_chunks)} raw chunks → "
            f"{len(unique_chunks)} unique after dedup"
        )

        # Step 4: Re-rank and return top_k_final
        reranked = self._reranker.rerank(
            query=query,           # rerank against original query, not variants
            chunks=unique_chunks,
            top_k=settings.top_k_final,
        )

        return reranked

    # ── Private ──────────────────────────────────────────────

    def _retrieve_multi(
        self,
        queries: List[str],
        top_k: int,
        filter_filename: Optional[str],
    ) -> List[RetrievedChunk]:
        """Run vector search for each query variant, collect all results."""
        all_chunks = []
        for q in queries:
            chunks = self._store.similarity_search(
                query=q,
                top_k=top_k,
                filter_filename=filter_filename,
            )
            all_chunks.extend(chunks)
        return all_chunks

    def _deduplicate(self, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """
        Remove duplicate chunks by chunk_id.
        When a chunk appears from multiple query variants,
        keep the instance with the highest similarity score.
        """
        seen: dict[str, RetrievedChunk] = {}
        for chunk in chunks:
            key = chunk.chunk_id
            if key not in seen or chunk.similarity_score > seen[key].similarity_score:
                seen[key] = chunk
        # Return sorted by score so reranker starts with best candidates
        return sorted(seen.values(), key=lambda c: c.similarity_score, reverse=True)
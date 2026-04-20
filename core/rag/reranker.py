# core/rag/reranker.py
# ============================================================
# Cross-Encoder Re-Ranker — Phase 2 intelligence layer.
#
# Why re-ranking matters:
#   Vector similarity (cosine/dot-product) measures how close
#   two embedding vectors are — it's fast but approximate.
#   A cross-encoder reads the QUERY and CHUNK TOGETHER and scores
#   true semantic relevance. It's slower but far more accurate.
#
# Two-stage retrieval (industry standard):
#   Stage 1 — Vector search: retrieve top 10 candidates fast (ChromaDB)
#   Stage 2 — Re-rank:       score all 10 with cross-encoder, keep top 5
#
# Model: cross-encoder/ms-marco-MiniLM-L-6-v2
#   - Free, local (sentence-transformers — already installed)
#   - 22M params, runs fast on CPU (~50ms for 10 candidates)
#   - Trained on MS MARCO passage ranking — great for Q&A over docs
#   - No API key, no cost, no rate limits
#
# Alternative: Cohere Rerank API (better quality, needs API key)
#   Uncomment _rerank_with_cohere() and set COHERE_API_KEY in .env
# ============================================================

from typing import List

from config.settings import get_settings
from models.schemas import RetrievedChunk
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class CrossEncoderReranker:
    """
    Re-ranks retrieved chunks using a local cross-encoder model.

    Usage:
        reranker = CrossEncoderReranker()
        top_chunks = reranker.rerank(query, chunks, top_k=5)
    """

    def __init__(self):
        self._model = self._load_model()

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = None,
    ) -> List[RetrievedChunk]:
        """
        Re-score chunks against the query, return top_k by relevance.
        Falls back to original order if model failed to load.
        """
        k = top_k or settings.top_k_final

        if not chunks:
            return chunks

        if self._model is None:
            logger.warning("⚠️  Reranker not loaded — using original vector scores.")
            return chunks[:k]

        # Build (query, passage) pairs for the cross-encoder
        pairs = [[query, chunk.content] for chunk in chunks]

        # Score all pairs in one batch
        scores = self._model.predict(pairs, show_progress_bar=False)

        # Zip scores back to chunks and sort descending
        scored = sorted(
            zip(scores, chunks),
            key=lambda x: x[0],
            reverse=True,
        )

        reranked = []
        for score, chunk in scored[:k]:
            # Update similarity_score to reflect reranker score for transparency
            chunk.similarity_score = round(float(score), 4)
            chunk.metadata["reranker_score"] = round(float(score), 4)
            chunk.metadata["reranked"] = True
            reranked.append(chunk)

        logger.info(
            f"🔀 Reranked {len(chunks)} → {len(reranked)} chunks | "
            f"top score: {reranked[0].similarity_score:.4f}"
        )
        return reranked

    # ── Private ──────────────────────────────────────────────

    def _load_model(self):
        """Load cross-encoder model. Returns None on failure (graceful)."""
        try:
            from sentence_transformers import CrossEncoder
            model_name = settings.reranker_model
            logger.info(f"🔌 Loading reranker: [bold]{model_name}[/bold]")
            model = CrossEncoder(
                model_name,
                max_length=512,    # truncate long passages
            )
            logger.info("  ✅ Reranker ready.")
            return model
        except Exception as e:
            logger.warning(f"⚠️  Failed to load reranker: {e}. Skipping reranking.")
            return None


# ── Optional: Cohere Rerank (better quality, needs API key) ──
#
# class CohereReranker:
#     def __init__(self):
#         import cohere
#         self._client = cohere.Client(settings.cohere_api_key)
#
#     def rerank(self, query, chunks, top_k=5):
#         docs = [c.content for c in chunks]
#         results = self._client.rerank(
#             query=query, documents=docs,
#             model="rerank-english-v3.0", top_n=top_k
#         )
#         reranked = []
#         for r in results.results:
#             chunk = chunks[r.index]
#             chunk.similarity_score = round(r.relevance_score, 4)
#             reranked.append(chunk)
#         return reranked
# core/rag/chain.py
# ============================================================
# RAG Chain — Phase 2: Redis cache + full retrieval pipeline.
#
# Flow:
#   QueryRequest
#     → Redis cache check (exact-match on normalized query)
#     → [cache MISS] QueryExpander → multi-query vector search
#     → Deduplicate → CrossEncoder rerank
#     → Format context + RAG prompt
#     → Groq LLM (LLaMA 3.3 70B)
#     → Redis cache set
#     → QueryResponse (answer + cited sources + cache flag)
# ============================================================

import time
from typing import Optional

from langchain_groq import ChatGroq

from config.settings import get_settings
from core.cache.redis_cache import RedisCache
from retrieval.prompts import RAG_PROMPT, format_context
from core.rag.retriever import Retriever
from core.vectorstore.chroma_store import ChromaVectorStore
from models.schemas import QueryRequest, QueryResponse
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


class RAGChain:
    """
    Orchestrates the full Phase 2 RAG pipeline.

    Usage:
        chain = RAGChain()
        response = chain.query(QueryRequest(question="What is my deductible?"))
        print(response.answer)
        print("from cache:", response.from_cache)
    """

    def __init__(self):
        self._vector_store = ChromaVectorStore()
        self._retriever = Retriever(self._vector_store)
        self._cache = RedisCache()
        self._llm = self._build_llm()
        self._chain = RAG_PROMPT | self._llm

        logger.info(
            f"🤖 RAG Chain ready | "
            f"LLM: [bold]{settings.groq_model}[/bold] | "
            f"Cache: {'✅ Redis' if self._cache.is_connected else '⚠️  disabled'} | "
            f"Reranker: ✅ cross-encoder | "
            f"QueryExpansion: {'✅ on' if settings.query_expansion_enabled else '⚠️  off'}"
        )

    def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute a full Phase 2 RAG query end-to-end.
        Returns QueryResponse with answer, sources, and cache metadata.
        """
        start = time.perf_counter()

        # Step 1: Redis cache check
        cached = self._cache.get(request.question)
        if cached:
            return cached

        # Step 2: Retrieve (expansion → multi-search → dedup → rerank)
        logger.info(f"🔍 Query: '{request.question[:80]}'")
        chunks = self._retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            filter_filename=request.filter_filename,
        )

        if not chunks:
            logger.warning("⚠️  No relevant chunks found.")

        # Step 3: Format context
        context = format_context(chunks)

        # Step 4: Generate answer
        logger.info(f"⚡ Generating via Groq ({settings.groq_model})...")
        ai_message = self._chain.invoke(
            {"context": context, "question": request.question}
        )

        duration = time.perf_counter() - start

        response = QueryResponse(
            question=request.question,
            answer=ai_message.content,
            sources=chunks,
            model_used=settings.groq_model,
            retrieval_count=len(chunks),
            duration_seconds=round(duration, 3),
            from_cache=False,
        )

        # Step 5: Cache result
        self._cache.set(request.question, response)

        logger.info(
            f"✅ Done in {duration:.2f}s | "
            f"{len(chunks)} sources | "
            f"{len(ai_message.content)} chars"
        )
        return response

    # ── Private ──────────────────────────────────────────────

    def _build_llm(self) -> ChatGroq:
        return ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0,
            max_tokens=2048,
        )
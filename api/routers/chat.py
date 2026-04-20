# api/routers/chat.py
# ============================================================
# Chat endpoints:
#
#   POST /chat/query    Standard JSON response (full answer at once)
#   POST /chat/stream   SSE streaming response (token-by-token)
#
# Streaming uses Server-Sent Events (SSE) — the frontend
# receives tokens incrementally as the LLM generates them,
# giving a ChatGPT-like typewriter experience.
#
# SSE event types emitted:
#   {"type": "token",   "content": "..."}   — LLM output token
#   {"type": "sources", "sources": [...]}   — cited chunks
#   {"type": "meta",    "metadata": {...}}  — timing, model info
#   {"type": "done"}                        — stream complete
#   {"type": "error",   "content": "..."}  — error occurred
# ============================================================

import json
import time
from typing import AsyncGenerator

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from langchain_groq import ChatGroq

from config.settings import get_settings
from core.cache.redis_cache import RedisCache
from retrieval.chain import RAGChain
from retrieval.chain import RAG_PROMPT, format_context
from core.rag.retriever import Retriever
from core.vectorstore.chroma_store import ChromaVectorStore
from models.schemas import QueryRequest, QueryResponse, StreamChunk
from utils.logger import get_logger

router = APIRouter(prefix="/chat", tags=["Chat"])
logger = get_logger(__name__)
settings = get_settings()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Ask a question (standard response)",
    description="Full RAG pipeline: retrieve → rerank → generate. Returns complete answer.",
)
async def query(request: QueryRequest) -> QueryResponse:
    try:
        chain = RAGChain()
        return chain.query(request)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.post(
    "/stream",
    summary="Ask a question (streaming SSE response)",
    description=(
        "Same RAG pipeline as /query but streams LLM tokens via Server-Sent Events. "
        "Sources are emitted after all tokens. Connect with EventSource or fetch+ReadableStream."
    ),
    response_class=StreamingResponse,
)
async def stream_query(request: QueryRequest):
    return StreamingResponse(
        _stream_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",        # disable nginx buffering
            "Access-Control-Allow-Origin": "*",
        },
    )


# ── SSE Generator ─────────────────────────────────────────────

async def _stream_generator(request: QueryRequest) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted strings.
    Each line is: data: {json}\n\n
    """
    def sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    start = time.perf_counter()

    try:
        # Step 1: Check cache — if hit, stream the cached answer token by token
        cache = RedisCache()
        cached = cache.get(request.question)
        if cached:
            for word in cached.answer.split(" "):
                yield sse({"type": "token", "content": word + " "})
            yield sse({"type": "sources", "sources": [s.model_dump() for s in cached.sources]})
            yield sse({"type": "meta", "metadata": {
                "from_cache": True,
                "duration_seconds": round(time.perf_counter() - start, 3),
                "model_used": cached.model_used,
            }})
            yield sse({"type": "done"})
            return

        # Step 2: Retrieve + rerank
        store = ChromaVectorStore()
        retriever = Retriever(store)
        chunks = retriever.retrieve(
            query=request.question,
            top_k=request.top_k,
            filter_filename=request.filter_filename,
        )
        context = format_context(chunks)

        # Step 3: Stream LLM tokens
        llm = ChatGroq(
            groq_api_key=settings.groq_api_key,
            model_name=settings.groq_model,
            temperature=0,
            max_tokens=2048,
            streaming=True,
        )
        chain = RAG_PROMPT | llm
        full_answer = ""

        async for chunk in chain.astream(
            {"context": context, "question": request.question}
        ):
            token = chunk.content
            if token:
                full_answer += token
                yield sse({"type": "token", "content": token})

        duration = round(time.perf_counter() - start, 3)

        # Step 4: Emit sources + metadata
        yield sse({"type": "sources", "sources": [s.model_dump() for s in chunks]})
        yield sse({"type": "meta", "metadata": {
            "from_cache": False,
            "duration_seconds": duration,
            "model_used": settings.groq_model,
            "retrieval_count": len(chunks),
        }})
        yield sse({"type": "done"})

        # Step 5: Cache the full response
        response = QueryResponse(
            question=request.question,
            answer=full_answer,
            sources=chunks,
            model_used=settings.groq_model,
            retrieval_count=len(chunks),
            duration_seconds=duration,
            from_cache=False,
        )
        cache.set(request.question, response)

    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield sse({"type": "error", "content": str(e)})
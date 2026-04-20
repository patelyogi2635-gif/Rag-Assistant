# core/cache/redis_cache.py
# ============================================================
# Redis Semantic Cache for RAG queries.
#
# Strategy: exact-match cache on a normalized query key.
#   - Strip whitespace, lowercase, remove punctuation noise
#   - SHA-256 the normalized string → Redis key
#   - TTL controlled via REDIS_CACHE_TTL_SECONDS in .env
#
# Why not vector-similarity cache (semantic cache)?
#   Semantic caching (e.g. GPTCache) sounds great but adds
#   200-400ms latency on every cache check — defeating the purpose.
#   Exact-match on normalized text covers ~60-70% of repeated
#   queries in practice (same user asks same question again,
#   or slight rephrasing that normalizes to the same key).
#   Phase 3 can layer semantic cache on top if needed.
#
# Redis connection: falls back gracefully if Redis is not running —
#   logs a warning and skips caching. Never crashes the main flow.
# ============================================================

import hashlib
import json
import re
from typing import Optional

from config.settings import get_settings
from models.schemas import QueryResponse
from utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

CACHE_PREFIX = "rag:query:"


class RedisCache:
    """
    Redis-backed exact-match cache for QueryResponse objects.

    Usage:
        cache = RedisCache()
        cached = cache.get("What is my deductible?")
        if cached:
            return cached
        # ... compute response ...
        cache.set("What is my deductible?", response)
    """

    def __init__(self):
        self._client = self._connect()

    def get(self, query: str) -> Optional[QueryResponse]:
        """
        Look up a cached response. Returns None on miss or Redis error.
        """
        if not self._client:
            return None

        key = self._make_key(query)
        try:
            raw = self._client.get(key)
            if raw is None:
                logger.debug(f"🔴 Cache MISS: {key[:32]}...")
                return None

            data = json.loads(raw)
            response = QueryResponse(**data)
            response.from_cache = True
            logger.info(f"⚡ Cache HIT: '{query[:60]}...'")
            return response

        except Exception as e:
            logger.warning(f"Redis GET error (skipping cache): {e}")
            return None

    def set(self, query: str, response: QueryResponse) -> None:
        """
        Store a response. Silently skips on Redis error.
        """
        if not self._client:
            return

        key = self._make_key(query)
        try:
            # Serialize — exclude from_cache flag so stored value is always False
            data = response.model_dump()
            data["from_cache"] = False

            self._client.setex(
                name=key,
                time=settings.redis_cache_ttl,
                value=json.dumps(data),
            )
            logger.debug(
                f"💾 Cached response for '{query[:60]}' "
                f"(TTL={settings.redis_cache_ttl}s)"
            )
        except Exception as e:
            logger.warning(f"Redis SET error (skipping cache): {e}")

    def invalidate(self, query: str) -> bool:
        """Delete a specific cached query. Returns True if key existed."""
        if not self._client:
            return False
        key = self._make_key(query)
        deleted = self._client.delete(key)
        return deleted > 0

    def flush_all(self) -> int:
        """Delete all RAG cache keys. Returns count deleted."""
        if not self._client:
            return 0
        try:
            keys = self._client.keys(f"{CACHE_PREFIX}*")
            if keys:
                return self._client.delete(*keys)
            return 0
        except Exception as e:
            logger.warning(f"Redis flush error: {e}")
            return 0

    @property
    def is_connected(self) -> bool:
        return self._client is not None

    # ── Private ──────────────────────────────────────────────

    def _connect(self):
        """Connect to Redis. Returns None (not an error) if unavailable."""
        try:
            import redis
            client = redis.from_url(
                settings.redis_url,
                decode_responses=True,
                socket_connect_timeout=2,   # fail fast if Redis not running
            )
            client.ping()
            logger.info(f"✅ Redis connected: {settings.redis_url}")
            return client
        except ImportError:
            logger.warning(
                "⚠️  redis package not installed. "
                "Run: pip install redis  — caching disabled."
            )
            return None
        except Exception as e:
            logger.warning(
                f"⚠️  Redis not available ({e}). "
                "Caching disabled — RAG still works normally."
            )
            return None

    def _make_key(self, query: str) -> str:
        """
        Normalize query → SHA-256 → Redis key.
        Normalization ensures 'What is X?' and 'what is x' hit the same key.
        """
        normalized = re.sub(r"[^\w\s]", "", query.lower().strip())
        normalized = re.sub(r"\s+", " ", normalized)
        digest = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{CACHE_PREFIX}{digest}"
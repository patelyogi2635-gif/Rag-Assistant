# ============================================================
# Dockerfile — RAG Assistant
# Multi-stage build: keeps final image lean (~700MB vs ~2.5GB)
# Models download at first boot and persist via Railway Volume
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        torch==2.4.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY . .

RUN mkdir -p data/uploads data/chroma_db data/sample_docs

# ── Model cache → Railway Volume ─────────────────────────────
# Mount a Railway Volume at /data so models persist across deploys.
# Set ALL HuggingFace / sentence-transformers cache dirs to /data/hf_cache
ENV HF_HOME=/data/hf_cache
ENV TRANSFORMERS_CACHE=/data/hf_cache
ENV SENTENCE_TRANSFORMERS_HOME=/data/hf_cache/sentence_transformers
# Disable the HF symlinks warning in non-writable envs
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

# ── Startup script ───────────────────────────────────────────
# Downloads models on first boot (persisted), then starts the server.
# On subsequent deploys the cache is warm → no re-download.
COPY start.sh /app/start.sh

RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chmod +x /app/start.sh

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Use start.sh instead of uvicorn directly
CMD ["/app/start.sh"]
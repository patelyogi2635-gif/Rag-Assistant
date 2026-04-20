# ============================================================
# Dockerfile — RAG Assistant
# Multi-stage build: keeps final image lean (~1.2GB vs ~3GB)
# ============================================================

# ── Stage 1: Builder ─────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ git \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps into /install
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir \
        torch==2.4.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# Runtime system deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/uploads data/chroma_db data/sample_docs

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Pre-download embedding + reranker models at build time
# so container starts instantly (no model download on first request)
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
print('Downloading BGE embedding model...'); \
SentenceTransformer('BAAI/bge-base-en-v1.5'); \
print('Downloading cross-encoder reranker...'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2'); \
print('Models cached.')"

EXPOSE 8000

# Health check — Railway uses this to know the container is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "1", "--timeout-keep-alive", "30"]
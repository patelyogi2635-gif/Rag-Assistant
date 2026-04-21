#!/bin/bash
set -e

# ── Warm model cache (skipped if already cached on the Volume) ──
echo "Checking model cache at $HF_HOME ..."

python -c "
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

cache = os.environ.get('SENTENCE_TRANSFORMERS_HOME', '/data/hf_cache')
print(f'Cache dir: {cache}')

print('Loading BGE embedding model...')
SentenceTransformer('BAAI/bge-base-en-v1.5')

print('Loading cross-encoder reranker...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print('Models ready.')
"

# ── Start the API server ────────────────────────────────────
echo "Starting API server..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 30
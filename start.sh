#!/bin/bash
set -e

echo "Checking model cache at $HF_HOME ..."

python -c "
import os
from sentence_transformers import SentenceTransformer, CrossEncoder

print('Loading BGE embedding model (ONNX backend)...')
SentenceTransformer(
    'BAAI/bge-base-en-v1.5',
    backend='onnx'          # <-- use ONNX instead of torch
)

print('Loading cross-encoder reranker...')
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

print('Models ready.')
"

echo "Starting API server..."
exec uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --timeout-keep-alive 30
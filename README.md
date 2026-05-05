<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=a8e063&height=120&section=header" width="100%"/>

<h1>
  <img src="https://readme-typing-svg.demolab.com?font=Instrument+Serif&size=42&pause=1000&color=A8E063&center=true&vCenter=true&width=600&lines=RAG+Assistant;Document+Intelligence+System" alt="RAG Assistant" />
</h1>

<p align="center">
  <strong>Ask anything about your documents. Get cited, accurate answers.</strong><br/>
  Built on a fully open-source stack — no OpenAI, no vendor lock-in.
</p>

<p align="center">
  <a href="#-quick-start"><img src="https://img.shields.io/badge/Quick%20Start-→-a8e063?style=for-the-badge&logoColor=white" /></a>
  &nbsp;
  <a href="#-api-reference"><img src="https://img.shields.io/badge/API%20Docs-→-6fa83c?style=for-the-badge" /></a>
  &nbsp;
  <a href="#-tech-stack"><img src="https://img.shields.io/badge/Tech%20Stack-→-4a7a2a?style=for-the-badge" /></a>
</p>

<br/>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?style=flat-square&logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=flat-square" />
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-F55036?style=flat-square" />
  <img src="https://img.shields.io/badge/ChromaDB-Vector_DB-FF6B35?style=flat-square" />
  <img src="https://img.shields.io/badge/Redis-Cache-DC382D?style=flat-square&logo=redis&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-Ready-2496ED?style=flat-square&logo=docker&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-22c55e?style=flat-square" />
</p>

<br/>

> Upload PDFs → Ask questions → Get answers with exact source citations.
> Medical records, legal contracts, research papers, policy documents — any domain.

<img src="https://capsule-render.vercel.app/api?type=rect&color=1c201c&height=2" width="100%"/>

</div>

<br/>

## 📌 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Docker Setup](#-docker-setup)
- [Environment Variables](#-environment-variables)
- [API Reference](#-api-reference)
- [How It Works](#-how-it-works)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧭 Overview

**RAG Assistant** is a production-grade Retrieval-Augmented Generation system that lets you have an accurate, cited conversation with your own documents.

Most AI chatbots hallucinate when answering questions about specific documents. RAG Assistant solves this by:

1. Breaking your PDFs into semantically meaningful chunks
2. Storing them in a vector database with rich metadata
3. At query time — retrieving the most relevant chunks using both vector similarity **and** neural re-ranking
4. Sending only those chunks to the LLM as grounded context

The result: answers that are accurate, fast, and always traceable to an exact page and document.

```
Without RAG  →  LLM guesses from training data         → hallucinations
With RAG     →  LLM reads your actual document chunks  → cited facts
```

---

## ✨ Features

<table>
<tr>
<td width="50%">

**📄 Document Management**
- Upload up to 5 PDFs simultaneously
- Supports any domain — medical, legal, finance, research
- SHA-256 deduplication — re-uploading same file is a no-op
- Scanned PDF detection with clear user feedback
- Per-file deletion from the vector store

</td>
<td width="50%">

**🔍 Intelligent Retrieval**
- Two-stage retrieval: vector search → cross-encoder reranking
- Query expansion: 3 LLM-generated alternative phrasings per query
- Deduplication across multi-query results by chunk ID
- Configurable `top_k_retrieval` and `top_k_final` parameters
- Optional filename-scoped search

</td>
</tr>
<tr>
<td width="50%">

**⚡ Performance**
- Redis exact-match semantic cache (sub-10ms on cache hit)
- ~500 tok/s generation via Groq inference
- Embedding model pre-loaded at startup (no cold-start lag)
- Persistent ChromaDB across restarts via Docker volumes

</td>
<td width="50%">

**🌊 Production API**
- FastAPI with full async support
- Server-Sent Events (SSE) for token-by-token streaming
- Auto-generated Swagger + ReDoc documentation
- CORS configured, health check endpoint
- Nginx reverse proxy with SSE-compatible config

</td>
</tr>
<tr>
<td width="50%">

**🎨 Clean Frontend**
- Zero build step — single HTML file
- Drag-and-drop PDF upload with progress
- Real-time typewriter streaming effect
- Source citations with page numbers and relevance scores
- Cache hit badges, timing metadata

</td>
<td width="50%">

**🔒 Open Source Stack**
- Zero OpenAI dependency
- Embeddings run 100% locally (BAAI/bge-base-en-v1.5)
- Reranker runs 100% locally (cross-encoder/ms-marco)
- Only external call: Groq API for LLM generation (free tier)
- Docker Compose for full local production parity

</td>
</tr>
</table>

---

## 🏗️ System Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG ASSISTANT                                   │
│                                                                               │
│  ┌─────────────┐    ┌──────────────────────────────┐    ┌──────────────────┐ │
│  │   Frontend  │    │       FastAPI Backend          │    │   Data Layer     │ │
│  │             │    │                                │    │                  │ │
│  │  index.html │◄──►│  POST /documents/upload        │◄──►│  ChromaDB        │ │
│  │  Dark UI    │    │  GET  /documents/              │    │  (Vector Store)  │ │
│  │  SSE Stream │    │  POST /chat/stream  ───────────┼──► │  Redis Cache     │ │
│  │  Citations  │    │  GET  /health                  │    │  HF Models       │ │
│  └─────────────┘    └──────────────────────────────┘    └──────────────────┘ │
│                                          │                                    │
│                                          ▼                                    │
│                                   Groq API                                    │
│                              (LLaMA 3.3 70B)                                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Ingestion Pipeline

```
PDF Upload (max 5 · 20MB each)
         │
         ▼
  ┌─────────────┐     Validates: extension · size · magic bytes
  │  Validation  │◄──  SHA-256 hash → skip if already indexed
  └──────┬──────┘
         │
         ▼
  ┌─────────────┐     pdfplumber (primary)
  │  PDF Loader  │◄──  pypdf (fallback)
  └──────┬──────┘     Scanned PDF detection + warning
         │
         ▼
  ┌─────────────┐     RecursiveCharacterTextSplitter
  │   Chunker    │◄──  Token-based: 1000 tok · 200 overlap
  └──────┬──────┘     tiktoken for accurate token counting
         │
         ▼
  ┌─────────────┐     BAAI/bge-base-en-v1.5 (local · free)
  │  Embedder   │◄──  BGE query prefix applied at query time
  └──────┬──────┘     Normalized embeddings (cosine sim ready)
         │
         ▼
  ┌─────────────┐     Persistent storage
  │  ChromaDB   │◄──  Metadata: source · page · hash · index
  └─────────────┘
```

### Query Pipeline

```
User Question
      │
      ▼
┌───────────┐
│   Redis   │──── HIT ──────────────────────────────────► Return instantly ⚡
│   Cache   │
└─────┬─────┘
      │ MISS
      ▼
┌───────────────┐
│ QueryExpander │  Groq generates 3 alternative phrasings
│               │  ["original q", "rephrasing 1", "rephrasing 2", "rephrasing 3"]
└──────┬────────┘
       │
       ▼
┌──────────────┐
│  ChromaDB    │  Similarity search × 4 queries
│  Vector      │  top_k_retrieval = 10 candidates per query
│  Search      │  → up to 40 raw candidates
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Deduplicate  │  Merge by chunk_id · keep highest score per chunk
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ CrossEncoder │  cross-encoder/ms-marco-MiniLM-L-6-v2 (local · free)
│  Reranker    │  Scores (query, chunk) pairs together
│              │  → top_k_final = 5 truly relevant chunks
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Groq LLM    │  LLaMA 3.3 70B · temperature=0 · streaming
│  Generation  │  System prompt enforces citation · no hallucination
└──────┬───────┘
       │
       ▼
  SSE Token Stream → Browser (typewriter effect)
  Sources + Metadata emitted after generation
  Response stored in Redis cache (TTL = 1hr)
```

---

## 📁 Project Structure

```
rag-assistant/
│
├── 📂 api/                           # FastAPI application layer
│   ├── main.py                       # App factory · CORS · lifespan · static files
│   └── routers/
│       ├── documents.py              # Upload · list · delete endpoints
│       ├── chat.py                   # Query · SSE stream endpoints
│       └── health.py                 # Health check endpoint
│
├── 📂 core/                          # Business logic
│   ├── ingestion/
│   │   ├── pipeline.py               # Orchestrator: validate → load → chunk → store
│   │   ├── pdf_loader.py             # Text extraction · scanned PDF detection
│   │   ├── chunker.py                # Token-based semantic chunking
│   │   └── embedder.py               # BGEEmbeddings wrapper (query prefix)
│   │
│   ├── vectorstore/
│   │   └── chroma_store.py           # ChromaDB CRUD · dedup · similarity search
│   │
│   ├── cache/
│   │   └── redis_cache.py            # Normalize → SHA-256 → Redis · graceful fallback
│   │
│   └── rag/
│       ├── chain.py                  # Main orchestrator: cache → retrieve → generate
│       ├── retriever.py              # expand → multi-search → dedup → rerank
│       ├── reranker.py               # CrossEncoderReranker (local)
│       ├── query_expander.py         # LLM query expansion (3 variants)
│       └── prompts.py                # System prompt · RAG answer template
│
├── 📂 models/
│   └── schemas.py                    # All Pydantic models (request · response)
│
├── 📂 config/
│   └── settings.py                   # Centralized config via pydantic-settings
│
├── 📂 utils/
│   ├── file_utils.py                 # SHA-256 hash · magic bytes validation
│   └── logger.py                     # Rich structured logging
│
├── 📂 frontend/
│   └── index.html                    # Single-file UI · no build step
│
├── 📂 nginx/
│   └── nginx.conf                    # Reverse proxy · SSE-compatible config
│
├── main.py                           # CLI demo runner
├── setup_project.py                  # One-time setup script
├── Dockerfile                        # Production Docker image
├── docker-compose.yml                # Full stack: app + Redis + Nginx
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment variable template
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **LLM** | Groq · LLaMA 3.3 70B | ~500 tok/s · free tier · same LangChain interface as OpenAI |
| **Embeddings** | BAAI/bge-base-en-v1.5 | Top MTEB benchmark · 768-dim · runs locally · zero cost |
| **Re-ranker** | cross-encoder/ms-marco-MiniLM-L-6-v2 | True semantic relevance · 22M params · ~50ms for 10 chunks |
| **Vector DB** | ChromaDB | Persistent · metadata filtering · no infra needed |
| **Cache** | Redis 7 | Sub-10ms hits · LRU eviction · TTL per entry |
| **Framework** | FastAPI | Async · SSE streaming · auto Swagger docs |
| **Orchestration** | LangChain 0.3 | Composable chains · provider-agnostic |
| **Chunking** | langchain-text-splitters + tiktoken | Token-accurate splitting (not character-based) |
| **Frontend** | Vanilla HTML/CSS/JS | Zero dependencies · zero build step · SSE native |
| **Proxy** | Nginx Alpine | SSE-compatible buffering · upload size control |
| **Containers** | Docker + Compose | Full local production parity |

### Design Decisions

**Why no OpenAI?** Cost and vendor lock. BGE embeddings (free, local) consistently score in the top 5 on the MTEB leaderboard. The cross-encoder reranker closes any quality gap at zero cost.

**Why two-stage retrieval?** Pure vector search misses ~20% of relevant chunks due to embedding approximation. The cross-encoder reads query and chunk *together* and is significantly more accurate — but too slow to run on thousands of chunks. The solution: vector search narrows to 10–40 candidates fast, cross-encoder scores those accurately, top 5 go to the LLM.

**Why query expansion?** Domain-specific documents (medical, legal, policy) use terminology that may not match the user's phrasing. "What is my deductible?" should also find chunks that say "out-of-pocket threshold" or "policy cap". LLM-generated paraphrases close this vocabulary gap.

**Why token-based chunking?** LLMs have *token* limits, not character limits. Measuring chunk size in characters produces inconsistent results — a 1000-character chunk can be 200 or 600 tokens depending on content. tiktoken gives exact token counts.

---

## 🚀 Quick Start

### Prerequisites

| Requirement | Version | Notes |
|------------|---------|-------|
| Python | 3.11+ | Required |
| Git | any | Required |
| Groq API Key | — | [Free at console.groq.com](https://console.groq.com/keys) |
| Docker | optional | For Redis cache + full stack |

### Step 1 — Clone

```bash
git clone https://github.com/patelyogi2635/rag-assistant.git
cd rag-assistant
```

### Step 2 — Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

> ⚠️ **Important:** PyTorch must be installed first from the CPU-only index.
> Installing it via pip directly pulls the CUDA build which fails without a GPU.

```bash
# 1. Install PyTorch CPU-only (Windows/Linux/macOS)
pip install torch==2.4.0+cpu --index-url https://download.pytorch.org/whl/cpu

# 2. Install all other dependencies
pip install -r requirements.txt
```

### Step 4 — Configure

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

```env
GROQ_API_KEY=gsk_your_key_here
```

All other defaults work out of the box.

### Step 5 — Initialize

```bash
# Creates all __init__.py package files + data/ directories
python setup_project.py
```

### Step 6 — Start Redis (Optional)

```bash
docker run -d -p 6379:6379 redis:alpine
```

> Redis enables query caching. Without it, the system still works — caching is gracefully disabled.

### Step 7 — Launch

```bash
uvicorn api.main:app --reload --port 8000
```

Open **[http://localhost:8000/ui](http://localhost:8000/ui)** ✅

---

## 🐳 Docker Setup

Run the full production stack (API + Redis + Nginx) with one command:

```bash
# Configure environment
cp .env.example .env
# → Set GROQ_API_KEY in .env

# Build and start
docker compose up --build

# Run in background
docker compose up -d --build
```

| URL | Description |
|-----|-------------|
| `http://localhost/ui` | Frontend UI |
| `http://localhost/docs` | Swagger API docs |
| `http://localhost/health` | Health check |

```bash
# View logs
docker compose logs -f app

# Stop
docker compose down

# Stop and remove volumes (clears ChromaDB + Redis data)
docker compose down -v
```

> **Data persistence:** ChromaDB vectors, Redis cache, and uploaded PDFs are stored in named Docker volumes and survive container restarts.

---

## ⚙️ Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | **required** | Your Groq API key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | LLM model name |
| `EMBEDDING_MODEL` | `BAAI/bge-base-en-v1.5` | HuggingFace embedding model |
| `EMBEDDING_DEVICE` | `cpu` | `cpu` or `cuda` |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder model |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | ChromaDB storage path |
| `REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `REDIS_CACHE_TTL_SECONDS` | `3600` | Cache TTL (1 hour) |
| `QUERY_EXPANSION_ENABLED` | `true` | Enable LLM query expansion |
| `QUERY_EXPANSION_N` | `3` | Number of query variants |
| `TOP_K_RETRIEVAL` | `10` | Candidates from vector search |
| `TOP_K_FINAL` | `5` | Chunks after reranking → sent to LLM |
| `CHUNK_SIZE` | `1000` | Tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Token overlap between chunks |
| `MAX_PDF_FILES` | `5` | Max files per upload |
| `MAX_PDF_SIZE_MB` | `20` | Max file size in MB |
| `LOG_LEVEL` | `INFO` | Logging level |

---

## 📡 API Reference

### Documents

#### Upload PDFs
```http
POST /documents/upload
Content-Type: multipart/form-data
```
```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "files=@report.pdf" \
  -F "files=@policy.pdf"
```
```json
{
  "processed": [
    { "filename": "report.pdf", "total_pages": 12, "total_chunks": 34, "was_duplicate": false },
    { "filename": "policy.pdf", "total_pages": 8,  "total_chunks": 22, "was_duplicate": false }
  ],
  "total_chunks_added": 56,
  "duration_seconds": 9.2
}
```

#### List Indexed Files
```http
GET /documents/
```
```json
{
  "files": [
    { "filename": "report.pdf", "chunk_count": 34 },
    { "filename": "policy.pdf", "chunk_count": 22 }
  ],
  "total_files": 2,
  "total_chunks": 56
}
```

#### Delete a File
```http
DELETE /documents/{filename}
```

---

### Chat

#### Standard Query
```http
POST /chat/query
Content-Type: application/json
```
```bash
curl -X POST http://localhost:8000/chat/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings in the report?"}'
```
```json
{
  "question": "What are the key findings in the report?",
  "answer": "The key findings are... [1] report.pdf p.3 [2] report.pdf p.7",
  "sources": [
    {
      "source_file": "report.pdf",
      "page_number": 3,
      "similarity_score": 0.9124,
      "content": "..."
    }
  ],
  "model_used": "llama-3.3-70b-versatile",
  "retrieval_count": 5,
  "duration_seconds": 1.84,
  "from_cache": false
}
```

#### Streaming Query (SSE)
```http
POST /chat/stream
Content-Type: application/json
```
```bash
curl -X POST http://localhost:8000/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -d '{"question": "Summarize the policy document"}' \
  --no-buffer
```

SSE event types:
```
data: {"type": "token",   "content": "The "}
data: {"type": "token",   "content": "policy "}
data: {"type": "sources", "sources": [...]}
data: {"type": "meta",    "metadata": {"duration_seconds": 1.2, "from_cache": false}}
data: {"type": "done"}
```

#### Optional Query Parameters
```json
{
  "question": "What is my coverage limit?",
  "top_k": 8,
  "filter_filename": "policy.pdf"
}
```

---

### System

#### Health Check
```http
GET /health
```
```json
{
  "status": "ok",
  "vector_store_chunks": 56,
  "redis_connected": true,
  "groq_model": "llama-3.3-70b-versatile",
  "embedding_model": "BAAI/bge-base-en-v1.5"
}
```

Interactive docs available at **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 🔬 How It Works

### Why Re-ranking Changes Everything

Standard RAG uses only vector similarity — embeddings measure how "close" two texts are in vector space. This is fast but approximate.

```
Vector search alone:
  Query: "What is the deductible?"
  ✅ Returns: "The deductible is $500 per year"         (score: 0.91)
  ❌ Misses:  "Out-of-pocket maximum threshold: $500"   (score: 0.61)  ← same info, different words
```

The cross-encoder reads query and passage **together** in one forward pass, capturing true semantic relevance — not just surface similarity.

```
After re-ranking:
  ✅ "The deductible is $500 per year"                 (rerank score: 0.94)
  ✅ "Out-of-pocket maximum threshold: $500"           (rerank score: 0.89)  ← now retrieved
```

### Retrieval Quality at Each Stage

```
ChromaDB vector search   → 10–40 candidates  (fast, approximate)
         ↓
Deduplication            → unique chunks only
         ↓
CrossEncoder reranking   → top 5             (slow, accurate)
         ↓
LLM receives             → 5 highly relevant, deduplicated chunks
```

---

## 🗺️ Roadmap

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Complete | PDF ingestion · token chunking · ChromaDB · base RAG |
| **Phase 2** | ✅ Complete | Cross-encoder reranking · Redis cache · query expansion |
| **Phase 3** | ✅ Complete | FastAPI REST API · SSE streaming · dark UI |
| **Phase 4** | 🔄 Planned | OCR support for scanned/image-based PDFs |
| **Phase 5** | 🔄 Planned | Multi-user sessions · JWT auth · per-user document isolation |
| **Phase 6** | 🔄 Planned | Conversation memory · follow-up questions · chat history |
| **Phase 7** | 🔄 Planned | Evaluation pipeline (RAGAS metrics) · automated quality scoring |

---

## 🤝 Contributing

Contributions are welcome. Please follow this workflow:

```bash
# 1. Fork the repository

# 2. Create a feature branch
git checkout -b feat/your-feature-name

# 3. Make your changes and commit
git commit -m "feat: add your feature description"

# 4. Push and open a Pull Request
git push origin feat/your-feature-name
```

**Commit convention:** `feat:` · `fix:` · `docs:` · `refactor:` · `chore:`

---



<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=a8e063&height=100&section=footer" width="100%"/>

<p>
  <strong>Built by <a href="https://github.com/YOUR_USERNAME">Yogi Patel</a></strong><br/>
  <sub>If this project helped you, consider giving it a ⭐</sub>
</p>

</div>

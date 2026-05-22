# RAG Knowledge Base API

A production-ready **Retrieval-Augmented Generation (RAG)** backend that ingests PDF documents and Python source code, processes them into semantic chunks with embeddings, and exposes a query API for semantic search and AI-powered Q&A.

---

## Table of Contents

- [Architecture](#architecture)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Ingestion Script](#ingestion-script)
- [Database Schema](#database-schema)
- [Scaling Strategy & Trade-offs](#scaling-strategy--trade-offs)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     CLIENT / API CONSUMER                     │
└──────────────────────────┬───────────────────────────────────┘
                           │  HTTP / REST
┌──────────────────────────▼───────────────────────────────────┐
│                  FastAPI Application (port 8000)              │
│                                                               │
│   POST /api/v1/upload     POST /api/v1/query                  │
│   GET  /api/v1/files      GET  /api/v1/health                 │
│   GET  /api/v1/jobs/{id}  GET  /api/v1/metrics                │
└──────┬───────────────────────────┬────────────────────────────┘
       │                           │
       │  BackgroundTask           │  Query Pipeline
       ▼                           ▼
┌─────────────────┐     ┌──────────────────────────────────────┐
│ Ingestion        │     │  1. Embed query (Sentence Transformer)│
│ Pipeline         │     │  2. Vector search  (Qdrant Cloud)    │
│                  │     │  3. Fetch metadata (PostgreSQL)       │
│ 1. Upload file   │     │  4. Synthesize answer (Gemini)        │
│ 2. Extract text  │     │  5. Return results + citations        │
│    PDF → pymupdf4llm + Gemini Vision OCR                      │
│    .py → AST-aware chunking                                   │
│ 3. Chunk text    │                                            │
│ 4. Embed chunks  │                                            │
│    (Sentence Transformers, local)                             │
│ 5. Upsert Qdrant │                                            │
│ 6. Save to PG    │                                            │
└─────────────────┘                                            │
                                                               │
┌──────────────────────────────────────────────────────────────┘
│                        Storage Layer                          │
│                                                               │
│  ┌─────────────────────┐    ┌──────────────────────────────┐  │
│  │   PostgreSQL 15      │    │      Qdrant Cloud            │  │
│  │                      │    │                              │  │
│  │  files               │    │  collection:                 │  │
│  │  chunks              │    │  rag_knowledge_base          │  │
│  │  processing_jobs     │    │  384-dim cosine vectors      │  │
│  │  audit_log           │    │  payload indexes             │  │
│  └─────────────────────┘    └──────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

### PDF Extraction Strategy

The PDF processor tries three methods in order:

| Priority | Method | When used |
|----------|--------|-----------|
| 1 | **PyMuPDF text layer** | PDF has embedded text (digital PDFs) |
| 2 | **pymupdf4llm markdown** | Mixed content PDFs |
| 3 | **Gemini Vision OCR** | Fully scanned / image-only PDFs |

---

## Tech Stack

| Layer | Technology | Version | Notes |
|-------|-----------|---------|-------|
| API Framework | FastAPI | 0.111 | Async, auto-docs, BackgroundTasks |
| ORM | SQLAlchemy + Alembic | 2.0 | Async ORM, migrations |
| Relational DB | PostgreSQL | 15 | Metadata, chunks, jobs |
| Vector DB | **Qdrant Cloud** | — | 384-dim cosine, eu-west-2 AWS |
| Embeddings | **Sentence Transformers** | 3.0 | `all-MiniLM-L6-v2`, local, free |
| LLM | **Google Gemini** | 2.0-flash | RAG synthesis + Vision OCR |
| PDF Parsing | PyMuPDF + pymupdf4llm | 1.24 | Text + markdown extraction |
| Containerisation | Docker + Compose | — | PostgreSQL + API |

> **No Redis. No Celery.** File processing runs as FastAPI `BackgroundTasks` — simpler, fewer moving parts, zero extra infrastructure.

---

## Project Structure

```
rag-knowledge-base/
│
├── app/
│   ├── api/v1/
│   │   ├── endpoints/
│   │   │   ├── upload.py        # POST /upload — file ingestion
│   │   │   ├── query.py         # POST /query  — semantic search + RAG
│   │   │   ├── files.py         # GET/DELETE /files — file management
│   │   │   └── health.py        # GET /health, /metrics
│   │   └── router.py
│   │
│   ├── core/
│   │   ├── config.py            # Settings (pydantic-settings + YAML)
│   │   ├── logging.py           # Structured JSON logging (structlog)
│   │   └── security.py          # API key auth + rate limiting
│   │
│   ├── db/
│   │   ├── models/              # SQLAlchemy ORM models
│   │   │   ├── file.py          # File — uploaded file record
│   │   │   ├── chunk.py         # Chunk — text segment + embedding ref
│   │   │   ├── job.py           # ProcessingJob — ingestion status
│   │   │   └── audit.py         # AuditLog — immutable operation log
│   │   ├── base.py              # Declarative base + mixins
│   │   └── session.py           # Async session factory
│   │
│   ├── services/
│   │   ├── ingestion/
│   │   │   ├── pdf_processor.py # PDF → text (pymupdf4llm + Gemini OCR)
│   │   │   ├── code_processor.py# Python → AST-aware chunks
│   │   │   └── chunker.py       # Sliding-window text chunker
│   │   ├── embedding/
│   │   │   └── embedder.py      # Sentence Transformers (local)
│   │   ├── vector_store/
│   │   │   └── qdrant_store.py  # Qdrant Cloud REST client
│   │   └── query/
│   │       └── rag_pipeline.py  # Embed → search → enrich → Gemini
│   │
│   ├── tasks/
│   │   └── processing.py        # BackgroundTask ingestion pipeline
│   │
│   ├── schemas/                 # Pydantic request/response models
│   ├── config.yaml              # Central config (DB, Qdrant, LLM, etc.)
│   └── main.py                  # FastAPI app factory + lifespan
│
├── alembic/                     # Database migrations
│   └── versions/
│       └── 001_initial_schema.py
│
├── scripts/
│   └── ingest_test_files.py     # Standalone ingestion script
│
├── tests/
│   ├── test_upload.py
│   ├── test_query.py
│   ├── test_chunker.py
│   └── test_health.py
│
├── input document/              # Task files for ingestion demo
│   ├── Knowledge_Base_Sample.pdf
│   └── Source_Code_Sample.py
│
├── .env.example                 # Environment variable template
├── app/config.yaml              # Structured config defaults
├── docker-compose.yml           # PostgreSQL + API
├── Dockerfile
└── requirements.txt
```

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- [Google Gemini API key](https://aistudio.google.com/app/apikey) (for LLM synthesis + OCR)
- [Qdrant Cloud](https://cloud.qdrant.io) cluster (free tier available)

### 1. Clone and configure

```bash
git clone <repo-url>
cd rag-knowledge-base
cp .env.example .env
```

Edit `.env` and fill in:

```env
GEMINI_API_KEY=your-gemini-api-key
QDRANT_URL=https://<cluster-id>.<region>.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your-qdrant-api-key
```

### 2. Start services

```bash
docker-compose up -d
```

This starts PostgreSQL and the FastAPI app. Qdrant runs in the cloud — no local container needed.

### 3. Run database migrations

```bash
docker-compose exec api alembic upgrade head
```

### 4. Verify health

```bash
curl http://localhost:8000/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "postgresql": {"status": "healthy", "latency_ms": 2.1},
    "qdrant":     {"status": "healthy", "latency_ms": 45.3}
  }
}
```

### 5. Ingest the task files

```bash
python scripts/ingest_test_files.py
```

This runs the full pipeline for both task files and prints a summary:

```
  File                                   Type     Method                  Chunks
  ─────────────────────────────────────────────────────────────────────────────
  Knowledge_Base_Sample.pdf              pdf      gemini_vision_ocr           22
  Source_Code_Sample.py                  python   ast_chunking                13
  ─────────────────────────────────────────────────────────────────────────────
  TOTAL vectors in Qdrant                                                     35
```

### 6. Query the knowledge base

```bash
# Semantic search only
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main system design requirements?",
    "top_k": 5
  }'

# With Gemini answer synthesis
curl -X POST http://localhost:8000/api/v1/query \
  -H "X-API-Key: dev-api-key-change-in-production" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What functions are defined in the source code?",
    "top_k": 5,
    "use_llm": true
  }'
```

---

## Configuration

Configuration is layered — **env vars always win** over `app/config.yaml` defaults.

### `app/config.yaml` — structured defaults

```yaml
vector_store:
  url: "https://<cluster>.qdrant.io:6333"
  collection_name: "rag_knowledge_base"
  distance_metric: "cosine"

embedding:
  provider: "sentence_transformer"
  model: "all-MiniLM-L6-v2"
  dimension: 384
  device: "cpu"
  normalize_embeddings: true

llm:
  provider: "gemini"
  model: "gemini-2.0-flash"
  max_tokens: 1024
  temperature: 0.1

chunking:
  size: 512        # tokens
  overlap: 50      # tokens
  min_size: 100    # tokens
```

### Key environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GEMINI_API_KEY` | — | **Required.** Gemini LLM + Vision OCR |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_API_KEY` | — | Qdrant Cloud API key |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL async URL |
| `API_KEY` | `dev-api-key-...` | API authentication key |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence Transformer model |
| `LLM_MODEL` | `gemini-2.0-flash` | Gemini model for synthesis |
| `CHUNK_SIZE` | `512` | Target chunk size (tokens) |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |
| `ENVIRONMENT` | `development` | `development` or `production` |

---

## API Reference

Interactive docs at `http://localhost:8000/docs` (Swagger UI) — disabled in production.

### Upload API

**`POST /api/v1/upload`**

Upload a file for ingestion. Processing runs in the background.

```bash
curl -X POST http://localhost:8000/api/v1/upload \
  -H "X-API-Key: <your-api-key>" \
  -F "file=@document.pdf" \
  -F 'metadata={"source": "docs", "tags": ["design"]}'
```

Response `202 Accepted`:
```json
{
  "file_id": "550e8400-e29b-41d4-a716-446655440000",
  "job_id":  "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
  "filename": "document.pdf",
  "file_size": 14236672,
  "file_type": "pdf",
  "status": "pending",
  "message": "File uploaded. Processing started in the background."
}
```

### Query API

**`POST /api/v1/query`**

Semantic search with optional Gemini answer synthesis.

```json
{
  "query":    "What are the API rate limits?",
  "top_k":    5,
  "use_llm":  true,
  "file_ids": ["550e8400-..."],
  "filters":  {"file_type": "pdf"},
  "min_score": 0.3
}
```

Response:
```json
{
  "query": "What are the API rate limits?",
  "answer": "According to [Source 1], the API allows 100 requests per 60 seconds...",
  "sources": [
    {
      "chunk_id":      "uuid",
      "file_id":       "uuid",
      "filename":      "document.pdf",
      "content":       "Rate limiting: 100 req/min per API key...",
      "score":         0.87,
      "page_number":   4,
      "function_name": null
    }
  ],
  "total_results": 5,
  "model_used": "gemini-2.0-flash",
  "cached": false
}
```

### File Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/files` | GET | List all files (paginated) |
| `/api/v1/files/{id}` | GET | File details + chunk summaries |
| `/api/v1/files/{id}` | DELETE | Soft-delete file + vectors |
| `/api/v1/jobs/{id}` | GET | Processing job status |
| `/api/v1/health` | GET | System health check |
| `/api/v1/metrics` | GET | Processing statistics |

---

## Ingestion Script

`scripts/ingest_test_files.py` is a standalone script that ingests the two task files directly into Qdrant Cloud without needing the API server running.

```bash
python scripts/ingest_test_files.py
```

It uses the same `PDFProcessor` and `CodeProcessor` as the API, so results are identical. The script prints file IDs at the end — use them in `/api/v1/query` with the `file_ids` filter to restrict search to a specific file.

---

## Database Schema

```sql
-- Tracks every uploaded file
files (
  id              UUID PRIMARY KEY,
  filename        VARCHAR(500),
  original_filename VARCHAR(500),
  file_type       VARCHAR(50),        -- 'pdf' | 'python' | 'text' ...
  file_size       BIGINT,
  sha256_hash     VARCHAR(64) UNIQUE, -- deduplication
  status          VARCHAR(50),        -- pending | processing | completed | failed
  chunk_count     INTEGER,
  storage_path    VARCHAR(1000),
  metadata        JSONB,
  created_at      TIMESTAMPTZ,
  updated_at      TIMESTAMPTZ,
  deleted_at      TIMESTAMPTZ         -- soft delete
)

-- One row per text chunk; embedding_id links to Qdrant point
chunks (
  id              UUID PRIMARY KEY,
  file_id         UUID REFERENCES files(id),
  content         TEXT,
  chunk_index     INTEGER,
  token_count     INTEGER,
  page_number     INTEGER,            -- PDFs
  start_line      INTEGER,            -- code files
  end_line        INTEGER,
  function_name   VARCHAR(500),       -- code files
  class_name      VARCHAR(500),       -- code files
  embedding_id    VARCHAR(255),       -- Qdrant point ID
  metadata        JSONB,
  created_at      TIMESTAMPTZ,
  deleted_at      TIMESTAMPTZ
)

-- Tracks background processing jobs
processing_jobs (
  id              UUID PRIMARY KEY,
  file_id         UUID REFERENCES files(id),
  status          VARCHAR(50),        -- queued | running | completed | failed
  progress        INTEGER,            -- 0–100
  current_step    VARCHAR(255),
  error_message   TEXT,
  started_at      TIMESTAMPTZ,
  completed_at    TIMESTAMPTZ,
  result          JSONB,
  created_at      TIMESTAMPTZ
)

-- Immutable audit trail
audit_log (
  id              UUID PRIMARY KEY,
  action          VARCHAR(100),
  entity_type     VARCHAR(100),
  entity_id       UUID,
  api_key_hash    VARCHAR(64),
  ip_address      VARCHAR(45),
  details         JSONB,
  created_at      TIMESTAMPTZ
)
```

---

## Scaling Strategy & Trade-offs

### Horizontal Scaling

**API layer** — FastAPI is stateless. Scale with multiple replicas behind a load balancer (Nginx / AWS ALB). Use Gunicorn with `--workers` for multi-process.

**Background processing** — currently runs as `FastAPI BackgroundTasks` (in-process). For high-throughput production workloads, migrate to a dedicated task queue (Celery + Redis or AWS SQS) by swapping `app/tasks/processing.py`.

**Vector DB** — Qdrant Cloud scales automatically. For self-hosted scale-out, use Qdrant's distributed mode with multiple shards.

**Relational DB** — PostgreSQL with read replicas for query-heavy workloads. Use PgBouncer for connection pooling.

### Trade-offs

| Decision | Trade-off |
|----------|-----------|
| **BackgroundTasks** (no Celery) | Simpler stack, fewer services — but tasks share the API process memory and can't be retried across restarts |
| **Sentence Transformers** (local) | Free, no API cost, fast — but lower quality than cloud embeddings for complex queries |
| **Qdrant Cloud** | Managed, scalable, no ops — but adds network latency vs. local Qdrant |
| **Gemini Vision OCR** | Handles scanned PDFs without Tesseract — but costs API credits per page |
| **Soft deletes** | Full audit trail — but requires periodic cleanup of deleted rows |
| **SHA-256 deduplication** | Prevents duplicate ingestion — but blocks re-ingestion of updated files with the same content |

### Production Checklist

- [ ] Replace `API_KEY` with a secrets manager (AWS Secrets Manager / Vault)
- [ ] Enable PostgreSQL SSL (`?sslmode=require` in `DATABASE_URL`)
- [ ] Set `ENVIRONMENT=production` to disable Swagger UI
- [ ] Configure log aggregation (CloudWatch / Datadog / ELK)
- [ ] Set up PostgreSQL automated backups
- [ ] Add HTTPS with TLS termination at the load balancer
- [ ] Tune `CHUNK_SIZE` and `CHUNK_OVERLAP` for your document domain
- [ ] Switch `EMBEDDING_MODEL` to `all-mpnet-base-v2` for higher quality (768 dims — update Qdrant collection dimension too)

Read me File


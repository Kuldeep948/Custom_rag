"""
scripts/ingest_test_files.py
────────────────────────────
Ingests the two task files into Qdrant Cloud:

  • input document/Knowledge_Base_Sample.pdf   → document RAG
  • input document/Source_Code_Sample.py       → code-base RAG

Pipeline per file:
  1. Extract / OCR  (PDFProcessor with Gemini Vision, CodeProcessor with AST)
  2. Chunk          (sliding-window with overlap)
  3. Embed          (sentence-transformers/all-MiniLM-L6-v2, local)
  4. Upsert         (Qdrant Cloud via REST — no SDK timeout issues)
  5. Verify         (semantic search smoke-test)

Run from the project root:
    python scripts/ingest_test_files.py
"""

import json
import os
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List

# ── Project root on sys.path ──────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

# ── Config (read from .env) ───────────────────────────────────────────────────
QDRANT_URL     = os.environ["QDRANT_URL"].rstrip("/")
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]
COLLECTION     = os.getenv("QDRANT_COLLECTION_NAME", "rag_knowledge_base")
EMBED_MODEL    = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
EMBED_DIM      = 384
BATCH_SIZE     = 64

INPUT_DIR = ROOT / "input document"
FILES = [
    {
        "path":      INPUT_DIR / "Knowledge_Base_Sample.pdf",
        "file_type": "pdf",
        "source":    "knowledge_base",
        "query":     "What are the main system design requirements?",
    },
    {
        "path":      INPUT_DIR / "Source_Code_Sample.py",
        "file_type": "python",
        "source":    "codebase",
        "query":     "What functions and classes are defined in this code?",
    },
]

# ── Colours ───────────────────────────────────────────────────────────────────
G = "\033[92m"; Y = "\033[93m"; C = "\033[96m"; R = "\033[91m"
B = "\033[1m";  X = "\033[0m"
def ok(m):   print(f"  {G}✓{X}  {m}")
def info(m): print(f"  {C}→{X}  {m}")
def warn(m): print(f"  {Y}⚠{X}  {m}")
def err(m):  print(f"  {R}✗{X}  {m}")
def hdr(m):  print(f"\n{B}{C}{m}{X}")
def sep():   print(f"  {'─'*60}")


# ═══════════════════════════════════════════════════════════════════════════════
# Qdrant REST helpers  (raw HTTP — avoids SDK timeout on Windows)
# ═══════════════════════════════════════════════════════════════════════════════

def _q(method: str, path: str, body: Any = None, timeout: int = 30) -> dict:
    url  = f"{QDRANT_URL}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req  = urllib.request.Request(
        url, data=data, method=method,
        headers={"api-key": QDRANT_API_KEY, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_collection() -> None:
    """Create collection + payload indexes if they don't exist yet."""
    try:
        res = _q("GET", f"/collections/{COLLECTION}")
        col = res["result"]
        ok(f"Collection '{COLLECTION}' exists — "
           f"status={col.get('status')}  "
           f"vectors={col.get('vectors_count') or 0}")
        return
    except Exception:
        pass

    info(f"Creating collection '{COLLECTION}' (dim={EMBED_DIM}, cosine) …")
    _q("PUT", f"/collections/{COLLECTION}", {
        "vectors": {"size": EMBED_DIM, "distance": "Cosine", "on_disk": False},
        "hnsw_config": {"m": 16, "ef_construct": 100},
        "optimizers_config": {"indexing_threshold": 20000},
    })
    for field in ("file_id", "file_type", "source"):
        _q("PUT", f"/collections/{COLLECTION}/index",
           {"field_name": field, "field_schema": "keyword"})
    ok(f"Collection '{COLLECTION}' created with payload indexes")


def upsert_points(points: List[dict]) -> None:
    for i in range(0, len(points), BATCH_SIZE):
        batch = points[i : i + BATCH_SIZE]
        _q("PUT", f"/collections/{COLLECTION}/points?wait=true",
           {"points": batch}, timeout=60)
        info(f"  Upserted {min(i + BATCH_SIZE, len(points))}/{len(points)} …")


def search(vector: List[float], file_id: str, limit: int = 3) -> List[dict]:
    res = _q("POST", f"/collections/{COLLECTION}/points/search", {
        "vector": vector,
        "limit": limit,
        "filter": {"must": [{"key": "file_id", "match": {"value": file_id}}]},
        "with_payload": True,
    }, timeout=20)
    return res.get("result", [])


def collection_count() -> int:
    res = _q("POST", f"/collections/{COLLECTION}/points/count", {"exact": True})
    return res.get("result", {}).get("count", 0)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    print(f"\n{B}{'═'*64}{X}")
    print(f"{B}  RAG Knowledge Base — Qdrant Cloud Ingestion{X}")
    print(f"{B}{'═'*64}{X}")
    print(f"  Collection : {C}{COLLECTION}{X}")
    print(f"  Cluster    : {C}{QDRANT_URL}{X}")
    print(f"  Embedder   : {C}{EMBED_MODEL}{X} ({EMBED_DIM} dims)")

    # ── 1. Collection ─────────────────────────────────────────────
    hdr("1 / 5  Qdrant Cloud — collection")
    ensure_collection()

    # ── 2. Embedding model ────────────────────────────────────────
    hdr("2 / 5  Loading Sentence Transformer")
    from sentence_transformers import SentenceTransformer
    info(f"Loading {EMBED_MODEL} …")
    model = SentenceTransformer(EMBED_MODEL)
    ok(f"Model ready — dim={model.get_sentence_embedding_dimension()}")

    # ── 3. Import app processors (uses project config) ────────────
    from app.services.ingestion.pdf_processor  import PDFProcessor
    from app.services.ingestion.code_processor import CodeProcessor

    summary = []

    for fi in FILES:
        path: Path = fi["path"]
        ftype: str = fi["file_type"]
        fid        = str(uuid.uuid4())

        hdr(f"3 / 5  Processing: {path.name}")
        sep()

        if not path.exists():
            err(f"File not found: {path}")
            err("Place task files in 'input document/' and re-run.")
            continue

        info(f"File   : {path}")
        info(f"Type   : {ftype}")
        info(f"FileID : {fid}")
        info(f"Size   : {path.stat().st_size / 1024:.1f} KB")

        # ── Extract + chunk ───────────────────────────────────────
        hdr("  ↳ Extracting & chunking")
        if ftype == "pdf":
            processor = PDFProcessor()
        else:
            processor = CodeProcessor()

        chunks, file_meta = processor.process(str(path))
        method = file_meta.get("extraction_method", "unknown")
        ok(f"Created {len(chunks)} chunks  (method: {method})")

        if not chunks:
            warn("No chunks produced — skipping file.")
            continue

        # ── Embed ─────────────────────────────────────────────────
        hdr("  ↳ Embedding")
        t0   = time.time()
        texts = [c.content for c in chunks]
        vecs  = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).tolist()
        ok(f"Embedded {len(vecs)} vectors in {time.time()-t0:.1f}s")

        # ── Upsert ────────────────────────────────────────────────
        hdr("  ↳ Upserting to Qdrant Cloud")
        points = [
            {
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "content":       chunk.content,
                    "file_id":       fid,
                    "filename":      path.name,
                    "file_type":     ftype,
                    "source":        fi["source"],
                    "chunk_index":   chunk.chunk_index,
                    "token_count":   chunk.token_count,
                    "chunk_type":    chunk.metadata.get("chunk_type", "text"),
                    "page_number":   chunk.page_number   or 0,
                    "start_line":    chunk.start_line    or 0,
                    "end_line":      chunk.end_line      or 0,
                    "function_name": chunk.function_name or "",
                    "class_name":    chunk.class_name    or "",
                },
            }
            for chunk, vec in zip(chunks, vecs)
        ]
        upsert_points(points)
        ok(f"Upserted {len(points)} vectors for '{path.name}'")

        # ── Verify ────────────────────────────────────────────────
        hdr("  ↳ Verification query")
        q_vec = model.encode(fi["query"], normalize_embeddings=True).tolist()
        hits  = search(q_vec, fid, limit=3)
        print(f"\n  {B}Query:{X} \"{fi['query']}\"")
        if not hits:
            warn("No results — check ingestion.")
        for i, h in enumerate(hits, 1):
            score   = h.get("score", 0)
            payload = h.get("payload", {})
            preview = payload.get("content", "")[:120].replace("\n", " ")
            fn      = payload.get("function_name", "")
            pg      = payload.get("page_number", 0)
            loc     = f"page {pg}" if pg else (f"fn:{fn}" if fn else "")
            print(f"  [{i}] score={score:.4f}  {loc}")
            print(f"      {C}{preview}…{X}")

        summary.append({
            "file": path.name, "file_id": fid,
            "type": ftype, "chunks": len(chunks),
            "method": method,
        })

    # ── 4. Collection stats ───────────────────────────────────────
    hdr("4 / 5  Collection Statistics")
    total = collection_count()
    ok(f"Total vectors in collection: {total}")

    # ── 5. Summary ────────────────────────────────────────────────
    hdr("5 / 5  Ingestion Summary")
    sep()
    print(f"  {'File':<38} {'Type':<8} {'Method':<22} {'Chunks':>6}")
    sep()
    for r in summary:
        print(f"  {r['file']:<38} {r['type']:<8} {r['method']:<22} {r['chunks']:>6}")
    sep()
    print(f"  {'TOTAL vectors in Qdrant':<48} {total:>6}")
    sep()

    print(f"\n  {G}{B}✓ Ingestion complete!{X}")
    print(f"  Dashboard : https://cloud.qdrant.io\n")
    print(f"  {B}File IDs (use in /api/v1/query → file_ids filter):{X}")
    for r in summary:
        print(f"    {r['file']:<38} {C}{r['file_id']}{X}")
    print()


if __name__ == "__main__":
    main()

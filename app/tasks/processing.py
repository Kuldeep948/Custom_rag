"""
File processing pipeline — runs as a FastAPI BackgroundTask.
No Celery, no Redis. Triggered directly after upload.

Pipeline: extract → chunk → embed → upsert to Qdrant → update PostgreSQL
"""
import asyncio
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.models.file import File, FileStatus
from app.db.models.job import JobStatus, ProcessingJob
from app.db.session import AsyncSessionLocal
from app.services.embedding.embedder import get_embedder_singleton
from app.services.ingestion.code_processor import CodeProcessor
from app.services.ingestion.pdf_processor import PDFProcessor
from app.services.vector_store.qdrant_store import get_vector_store

logger = get_logger(__name__)


async def process_file_background(file_id: str, job_id: str) -> None:
    """
    Full ingestion pipeline executed as a FastAPI BackgroundTask.

    Opens its own DB session (the request session is already closed by the
    time this runs) and processes the file end-to-end.

    Steps
    -----
    1. Load File + ProcessingJob records from PostgreSQL
    2. Extract text  (PDFProcessor or CodeProcessor)
    3. Chunk text    (sliding-window with overlap)
    4. Embed chunks  (SentenceTransformer, local)
    5. Upsert to Qdrant Cloud
    6. Save Chunk records to PostgreSQL
    7. Mark job + file as completed
    """
    logger.info("background_processing_start", file_id=file_id, job_id=job_id)

    async with AsyncSessionLocal() as session:
        try:
            await _run_pipeline(session, file_id, job_id)
        except Exception as exc:
            logger.error(
                "background_processing_failed",
                file_id=file_id,
                error=str(exc),
                exc_info=True,
            )
            await _mark_failed(session, file_id, job_id, str(exc))


async def _run_pipeline(
    session: AsyncSession, file_id: str, job_id: str
) -> None:
    """Core pipeline logic — raises on error so the caller can mark failure."""
    from sqlalchemy import select

    file = await session.get(File, uuid.UUID(file_id))
    job  = await session.get(ProcessingJob, uuid.UUID(job_id))

    if not file or not job:
        raise ValueError(f"File or job not found: file_id={file_id}, job_id={job_id}")

    # ── Step 1: Mark as running ───────────────────────────────────
    file.status       = FileStatus.PROCESSING
    job.status        = JobStatus.RUNNING
    job.started_at    = datetime.now(timezone.utc)
    job.current_step  = "Extracting text"
    job.progress      = 10
    await session.commit()

    # ── Step 2: Extract text ──────────────────────────────────────
    file_path = file.storage_path
    if not file_path or not Path(file_path).exists():
        raise FileNotFoundError(f"Uploaded file missing on disk: {file_path}")

    extension = (file.file_extension or "").lower()
    processor = PDFProcessor() if extension == ".pdf" else CodeProcessor()

    text_chunks, file_metadata = processor.process(file_path)

    if not text_chunks:
        raise ValueError("No content could be extracted from the file.")

    file.metadata_ = {**file.metadata_, **file_metadata}
    job.current_step = "Generating embeddings"
    job.progress     = 40
    await session.commit()

    # ── Step 3: Embed ─────────────────────────────────────────────
    embedder = get_embedder_singleton()
    texts    = [c.content for c in text_chunks]

    # SentenceTransformer runs in a thread-pool inside embed_texts
    embeddings = await embedder.embed_texts(texts)

    job.current_step = "Storing in Qdrant"
    job.progress     = 70
    await session.commit()

    # ── Step 4: Build records ─────────────────────────────────────
    chunk_ids:      list[str]        = []
    chunk_records:  list[Chunk]      = []
    vector_payloads: list[Dict[str, Any]] = []

    for text_chunk, embedding in zip(text_chunks, embeddings):
        cid = str(uuid.uuid4())
        chunk_ids.append(cid)

        chunk_records.append(
            Chunk(
                id=uuid.UUID(cid),
                file_id=file.id,
                content=text_chunk.content,
                chunk_index=text_chunk.chunk_index,
                token_count=text_chunk.token_count,
                page_number=text_chunk.page_number,
                start_line=text_chunk.start_line,
                end_line=text_chunk.end_line,
                function_name=text_chunk.function_name,
                class_name=text_chunk.class_name,
                embedding_id=cid,
                metadata_=text_chunk.metadata,
            )
        )

        vector_payloads.append({
            "file_id":       str(file.id),
            "filename":      file.original_filename,
            "file_type":     file.file_type or "",
            "chunk_index":   text_chunk.chunk_index,
            "page_number":   text_chunk.page_number or 0,
            "start_line":    text_chunk.start_line or 0,
            "end_line":      text_chunk.end_line or 0,
            "function_name": text_chunk.function_name or "",
            "class_name":    text_chunk.class_name or "",
        })

    # ── Step 5: Upsert to Qdrant ──────────────────────────────────
    vector_store = get_vector_store()
    vector_store.add_chunks(
        chunk_ids=chunk_ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=vector_payloads,
    )

    # ── Step 6: Save chunks to PostgreSQL ─────────────────────────
    session.add_all(chunk_records)

    # ── Step 7: Finalise ──────────────────────────────────────────
    file.status      = FileStatus.COMPLETED
    file.chunk_count = len(chunk_records)
    job.status       = JobStatus.COMPLETED
    job.progress     = 100
    job.current_step = "Complete"
    job.completed_at = datetime.now(timezone.utc)
    job.result       = {
        "chunk_count":   len(chunk_records),
        "file_metadata": file_metadata,
    }
    await session.commit()

    logger.info(
        "background_processing_complete",
        file_id=file_id,
        chunks=len(chunk_records),
        job_id=job_id,
    )


async def _mark_failed(
    session: AsyncSession, file_id: str, job_id: str, error: str
) -> None:
    """Mark file and job as failed in PostgreSQL."""
    try:
        file = await session.get(File, uuid.UUID(file_id))
        job  = await session.get(ProcessingJob, uuid.UUID(job_id))
        now  = datetime.now(timezone.utc)

        if file:
            file.status        = FileStatus.FAILED
            file.error_message = error
        if job:
            job.status         = JobStatus.FAILED
            job.error_message  = error
            job.completed_at   = now

        await session.commit()
    except Exception as e:
        logger.error("mark_failed_error", error=str(e))

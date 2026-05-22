"""
File management endpoints.
List, retrieve, and delete ingested files.
"""
import uuid
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.db.models.chunk import Chunk
from app.db.models.file import File, FileStatus
from app.db.models.job import ProcessingJob
from app.db.session import get_db
from app.schemas.file import (
    ChunkSummary,
    DeleteResponse,
    FileDetailResponse,
    FileListResponse,
    FileResponse,
    JobResponse,
)
from app.services.vector_store.qdrant_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/files",
    response_model=FileListResponse,
    summary="List all ingested files",
)
async def list_files(
    page: int = Query(default=1, ge=1, description="Page number"),
    page_size: int = Query(default=20, ge=1, le=100, description="Items per page"),
    status_filter: Optional[str] = Query(None, alias="status", description="Filter by status"),
    file_type: Optional[str] = Query(None, description="Filter by file type"),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> FileListResponse:
    """List all non-deleted files with pagination."""
    stmt = select(File).where(File.deleted_at.is_(None))

    if status_filter:
        stmt = stmt.where(File.status == status_filter)
    if file_type:
        stmt = stmt.where(File.file_type == file_type)

    # Count total
    count_stmt = select(func.count()).select_from(stmt.subquery())
    total = (await db.execute(count_stmt)).scalar_one()

    # Paginate
    offset = (page - 1) * page_size
    stmt = stmt.order_by(File.created_at.desc()).offset(offset).limit(page_size)
    files = (await db.execute(stmt)).scalars().all()

    return FileListResponse(
        items=[
            FileResponse(
                id=f.id,
                filename=f.filename,
                original_filename=f.original_filename,
                file_type=f.file_type,
                file_extension=f.file_extension,
                file_size=f.file_size,
                status=f.status,
                chunk_count=f.chunk_count,
                sha256_hash=f.sha256_hash,
                metadata=f.metadata_,
                error_message=f.error_message,
                created_at=f.created_at,
                updated_at=f.updated_at,
                deleted_at=f.deleted_at,
            )
            for f in files
        ],
        total=total,
        page=page,
        page_size=page_size,
        has_next=(offset + page_size) < total,
    )


@router.get(
    "/files/{file_id}",
    response_model=FileDetailResponse,
    summary="Get file details with chunks",
)
async def get_file(
    file_id: uuid.UUID,
    include_chunks: bool = Query(default=False, description="Include chunk summaries"),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> FileDetailResponse:
    """Get detailed information about a specific file."""
    stmt = select(File).where(File.id == file_id, File.deleted_at.is_(None))
    if include_chunks:
        stmt = stmt.options(selectinload(File.chunks))

    file = (await db.execute(stmt)).scalar_one_or_none()
    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_id} not found.",
        )

    chunk_summaries = []
    if include_chunks and file.chunks:
        for chunk in sorted(file.chunks, key=lambda c: c.chunk_index):
            if chunk.deleted_at is None:
                chunk_summaries.append(
                    ChunkSummary(
                        id=chunk.id,
                        chunk_index=chunk.chunk_index,
                        token_count=chunk.token_count,
                        page_number=chunk.page_number,
                        start_line=chunk.start_line,
                        end_line=chunk.end_line,
                        content_preview=chunk.content[:200],
                    )
                )

    return FileDetailResponse(
        id=file.id,
        filename=file.filename,
        original_filename=file.original_filename,
        file_type=file.file_type,
        file_extension=file.file_extension,
        file_size=file.file_size,
        status=file.status,
        chunk_count=file.chunk_count,
        sha256_hash=file.sha256_hash,
        metadata=file.metadata_,
        error_message=file.error_message,
        created_at=file.created_at,
        updated_at=file.updated_at,
        deleted_at=file.deleted_at,
        chunks=chunk_summaries,
    )


@router.delete(
    "/files/{file_id}",
    response_model=DeleteResponse,
    summary="Delete a file and its chunks",
)
async def delete_file(
    file_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> DeleteResponse:
    """
    Soft-delete a file and all its chunks.
    Also removes vectors from Qdrant.
    """
    file = (
        await db.execute(
            select(File).where(File.id == file_id, File.deleted_at.is_(None))
        )
    ).scalar_one_or_none()

    if not file:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"File {file_id} not found.",
        )

    now = datetime.now(timezone.utc)

    # Soft-delete the file
    file.deleted_at = now

    # Soft-delete all chunks
    chunks = (
        await db.execute(
            select(Chunk).where(Chunk.file_id == file_id, Chunk.deleted_at.is_(None))
        )
    ).scalars().all()

    for chunk in chunks:
        chunk.deleted_at = now

    await db.commit()

    # Remove from vector store
    try:
        vector_store = get_vector_store()
        deleted_count = vector_store.delete_by_file(str(file_id))
        logger.info(
            "file_deleted",
            file_id=str(file_id),
            chunks_deleted=len(chunks),
            vectors_deleted=deleted_count,
        )
    except Exception as e:
        logger.warning("vector_store_delete_failed", file_id=str(file_id), error=str(e))

    return DeleteResponse(
        success=True,
        message=f"File '{file.original_filename}' and {len(chunks)} chunks deleted.",
        file_id=file_id,
    )


@router.get(
    "/jobs/{job_id}",
    response_model=JobResponse,
    summary="Get processing job status",
)
async def get_job_status(
    job_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> JobResponse:
    """Check the status of a file processing job."""
    job = (
        await db.execute(select(ProcessingJob).where(ProcessingJob.id == job_id))
    ).scalar_one_or_none()

    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job {job_id} not found.",
        )

    return JobResponse(
        id=job.id,
        file_id=job.file_id,
        status=job.status,
        progress=job.progress,
        current_step=job.current_step,
        error_message=job.error_message,
        started_at=job.started_at,
        completed_at=job.completed_at,
        result=job.result,
        created_at=job.created_at,
    )

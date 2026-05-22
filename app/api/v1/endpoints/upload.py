"""
Upload API endpoint.
Validates → saves file → creates DB records → triggers BackgroundTask processing.
No Celery, no Redis.
"""
import hashlib
import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter, BackgroundTasks, Depends, File, Form,
    HTTPException, Request, UploadFile, status,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.db.models.file import File as FileModel, FileStatus, FileType
from app.db.models.job import JobStatus, ProcessingJob
from app.db.session import get_db
from app.schemas.upload import UploadResponse
from app.tasks.processing import process_file_background

logger = get_logger(__name__)
router = APIRouter()


def _detect_file_type(extension: str) -> str:
    mapping = {
        ".pdf": FileType.PDF,
        ".py":  FileType.PYTHON,
        ".pyw": FileType.PYTHON,
        ".txt": FileType.TEXT,
        ".md":  FileType.MARKDOWN,
    }
    return mapping.get(extension.lower(), FileType.CODE)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload and ingest a file",
    description=(
        "Upload a PDF or source code file. The file is saved immediately and "
        "processed in the background (extract → chunk → embed → Qdrant). "
        "Poll /api/v1/jobs/{job_id} to track progress."
    ),
)
async def upload_file(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="PDF or source code file"),
    metadata: Optional[str] = Form(
        None,
        description="Optional JSON metadata string, e.g. '{\"source\": \"docs\"}'",
    ),
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> UploadResponse:
    """
    Upload and ingest a file into the knowledge base.

    1. Validates file type and size
    2. Checks for duplicates via SHA-256 hash
    3. Saves file to disk
    4. Creates File + ProcessingJob records in PostgreSQL
    5. Schedules background processing (no Celery needed)
    """
    # ── Validate filename ─────────────────────────────────────────
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename is required.",
        )

    extension = Path(file.filename).suffix.lower()
    if extension.lstrip(".") not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"File type '{extension}' is not supported. "
                f"Allowed: {', '.join(settings.allowed_extensions_list)}"
            ),
        )

    # ── Read content ──────────────────────────────────────────────
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    if file_size > settings.max_file_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=(
                f"File size {file_size / 1024 / 1024:.1f} MB exceeds "
                f"the {settings.max_file_size_mb} MB limit."
            ),
        )

    # ── Deduplication ─────────────────────────────────────────────
    sha256_hash = hashlib.sha256(content).hexdigest()
    existing = (
        await db.execute(
            select(FileModel).where(
                FileModel.sha256_hash == sha256_hash,
                FileModel.deleted_at.is_(None),
            )
        )
    ).scalar_one_or_none()

    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={
                "message": "This file has already been ingested.",
                "existing_file_id": str(existing.id),
                "existing_status": existing.status,
            },
        )

    # ── Parse metadata ────────────────────────────────────────────
    parsed_metadata: dict = {}
    if metadata:
        try:
            parsed_metadata = json.loads(metadata)
            if not isinstance(parsed_metadata, dict):
                raise ValueError("Metadata must be a JSON object.")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid metadata JSON: {e}",
            )

    # ── Save to disk ──────────────────────────────────────────────
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_id = uuid.uuid4()
    safe_filename = f"{file_id}{extension}"
    storage_path = str(upload_dir / safe_filename)

    with open(storage_path, "wb") as f:
        f.write(content)

    logger.info(
        "file_saved",
        file_id=str(file_id),
        filename=file.filename,
        size=file_size,
    )

    # ── Create DB records ─────────────────────────────────────────
    file_type = _detect_file_type(extension)

    db_file = FileModel(
        id=file_id,
        filename=safe_filename,
        original_filename=file.filename,
        file_type=file_type,
        file_extension=extension,
        file_size=file_size,
        mime_type=file.content_type,
        sha256_hash=sha256_hash,
        status=FileStatus.PENDING,
        storage_path=storage_path,
        metadata_=parsed_metadata,
    )
    db.add(db_file)

    job_id = uuid.uuid4()
    db_job = ProcessingJob(
        id=job_id,
        file_id=file_id,
        status=JobStatus.QUEUED,
        progress=0,
        current_step="Queued for processing",
    )
    db.add(db_job)
    await db.commit()

    # ── Schedule background processing ───────────────────────────
    background_tasks.add_task(
        process_file_background,
        file_id=str(file_id),
        job_id=str(job_id),
    )

    logger.info(
        "processing_scheduled",
        file_id=str(file_id),
        job_id=str(job_id),
    )

    return UploadResponse(
        file_id=file_id,
        job_id=job_id,
        filename=file.filename,
        file_size=file_size,
        file_type=file_type,
        status=FileStatus.PENDING,
        message=(
            f"'{file.filename}' uploaded successfully. "
            "Processing started in the background. "
            "Poll /api/v1/jobs/{job_id} to track progress."
        ),
    )

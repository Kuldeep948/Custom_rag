"""
Pydantic schemas for file management endpoints.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ChunkSummary(BaseModel):
    """Brief summary of a chunk (used in file detail responses)."""

    id: uuid.UUID
    chunk_index: int
    token_count: Optional[int] = None
    page_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    content_preview: str = Field(..., description="First 200 chars of chunk content")

    model_config = {"from_attributes": True}


class FileResponse(BaseModel):
    """Full file details."""

    id: uuid.UUID
    filename: str
    original_filename: str
    file_type: Optional[str] = None
    file_extension: Optional[str] = None
    file_size: Optional[int] = None
    status: str
    chunk_count: int
    sha256_hash: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class FileDetailResponse(FileResponse):
    """File details including chunk summaries."""

    chunks: List[ChunkSummary] = Field(default_factory=list)


class FileListResponse(BaseModel):
    """Paginated list of files."""

    items: List[FileResponse]
    total: int
    page: int
    page_size: int
    has_next: bool


class JobResponse(BaseModel):
    """Processing job status."""

    id: uuid.UUID
    file_id: uuid.UUID
    status: str
    progress: int
    current_step: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class DeleteResponse(BaseModel):
    """Response for delete operations."""

    success: bool
    message: str
    file_id: uuid.UUID

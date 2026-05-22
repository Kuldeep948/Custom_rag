"""
Pydantic schemas for the Upload API.
"""
import uuid
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response returned immediately after a file upload."""

    file_id: uuid.UUID = Field(..., description="Unique identifier for the uploaded file")
    job_id: uuid.UUID = Field(..., description="Processing job ID to track progress")
    filename: str = Field(..., description="Original filename")
    file_size: int = Field(..., description="File size in bytes")
    file_type: str = Field(..., description="Detected file type (pdf, python, etc.)")
    status: str = Field(..., description="Initial processing status")
    message: str = Field(..., description="Human-readable status message")

    model_config = {"from_attributes": True}


class FileMetadataRequest(BaseModel):
    """Optional metadata to attach to an uploaded file."""

    source: Optional[str] = Field(None, description="Source system or category")
    tags: Optional[list[str]] = Field(default_factory=list, description="Searchable tags")
    description: Optional[str] = Field(None, description="Human-readable description")
    extra: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Additional key-value metadata"
    )

"""
File model — tracks all uploaded and ingested files.
"""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

from sqlalchemy import BigInteger, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.chunk import Chunk
    from app.db.models.job import ProcessingJob


class FileStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class FileType:
    PDF = "pdf"
    PYTHON = "python"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"


class File(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    Represents an uploaded file in the knowledge base.
    Tracks processing status, metadata, and relationships to chunks.
    """
    __tablename__ = "files"

    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    original_filename: Mapped[str] = mapped_column(String(500), nullable=False)
    file_type: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    file_extension: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    file_size: Mapped[Optional[int]] = mapped_column(BigInteger, nullable=True)
    mime_type: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # SHA-256 hash for deduplication
    sha256_hash: Mapped[Optional[str]] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )

    # Processing state
    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=FileStatus.PENDING, index=True
    )
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Storage path (relative to upload dir)
    storage_path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)

    # Flexible metadata (page count, language, etc.)
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    # Relationships
    chunks: Mapped[List["Chunk"]] = relationship(
        "Chunk",
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="select",
    )
    jobs: Mapped[List["ProcessingJob"]] = relationship(
        "ProcessingJob",
        back_populates="file",
        cascade="all, delete-orphan",
        lazy="select",
    )

    def __repr__(self) -> str:
        return f"<File id={self.id} filename={self.filename} status={self.status}>"

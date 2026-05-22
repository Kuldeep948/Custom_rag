"""
ProcessingJob model — tracks async file processing jobs.
"""
import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.file import File


class JobStatus:
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingJob(Base, UUIDMixin, TimestampMixin):
    """
    Tracks the status of an async file processing job.
    One job per file upload.
    """
    __tablename__ = "processing_jobs"

    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Celery task ID
    celery_task_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )

    status: Mapped[str] = mapped_column(
        String(50), nullable=False, default=JobStatus.QUEUED, index=True
    )

    # Progress percentage (0-100)
    progress: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # Step description (e.g., "Extracting text", "Generating embeddings")
    current_step: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)

    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    completed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    # Job result summary
    result: Mapped[Optional[dict]] = mapped_column(JSONB, nullable=True)

    # Relationship
    file: Mapped["File"] = relationship("File", back_populates="jobs")

    def __repr__(self) -> str:
        return (
            f"<ProcessingJob id={self.id} file_id={self.file_id} "
            f"status={self.status} progress={self.progress}%>"
        )

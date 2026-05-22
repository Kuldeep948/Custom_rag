"""
Chunk model — stores text segments extracted from files.
Each chunk has a corresponding vector embedding in Qdrant.
"""
import uuid
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, SoftDeleteMixin, TimestampMixin, UUIDMixin

if TYPE_CHECKING:
    from app.db.models.file import File


class Chunk(Base, UUIDMixin, TimestampMixin, SoftDeleteMixin):
    """
    A text chunk extracted from a file.
    The embedding_id links this chunk to its vector in Qdrant.
    """
    __tablename__ = "chunks"

    # Parent file
    file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("files.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Content
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Position metadata (for PDFs)
    page_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Position metadata (for code files)
    start_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    end_line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    function_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    class_name: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Link to vector store
    embedding_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True, index=True
    )

    # Additional metadata
    metadata_: Mapped[dict] = mapped_column(
        "metadata", JSONB, default=dict, nullable=False
    )

    # Relationship
    file: Mapped["File"] = relationship("File", back_populates="chunks")

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} file_id={self.file_id} "
            f"index={self.chunk_index} tokens={self.token_count}>"
        )

"""
Base text chunking utilities.
Implements token-aware sliding window chunking with overlap.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A single text chunk with positional metadata."""

    content: str
    chunk_index: int
    token_count: int = 0
    page_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    metadata: dict = field(default_factory=dict)


def estimate_tokens(text: str) -> int:
    """
    Fast token count estimate (4 chars ≈ 1 token).
    Use tiktoken for exact counts when needed.
    """
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    chunk_size: int = None,
    chunk_overlap: int = None,
    min_chunk_size: int = None,
    page_number: Optional[int] = None,
) -> List[TextChunk]:
    """
    Split text into overlapping chunks using a sliding window approach.
    Tries to split on sentence/paragraph boundaries when possible.

    Args:
        text: Input text to chunk
        chunk_size: Target chunk size in tokens
        chunk_overlap: Overlap between consecutive chunks in tokens
        min_chunk_size: Minimum chunk size (smaller chunks are discarded)
        page_number: Optional page number for PDF chunks

    Returns:
        List of TextChunk objects
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    min_chunk_size = min_chunk_size or settings.min_chunk_size

    if not text or not text.strip():
        return []

    # Convert token limits to approximate character limits
    char_size = chunk_size * 4
    char_overlap = chunk_overlap * 4
    min_char_size = min_chunk_size * 4

    # Split into sentences/paragraphs for cleaner boundaries
    paragraphs = _split_into_paragraphs(text)

    chunks: List[TextChunk] = []
    current_chunk = ""
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # If adding this paragraph would exceed chunk size, flush current chunk
        if current_chunk and len(current_chunk) + len(para) + 1 > char_size:
            if len(current_chunk) >= min_char_size:
                chunks.append(
                    TextChunk(
                        content=current_chunk.strip(),
                        chunk_index=chunk_index,
                        token_count=estimate_tokens(current_chunk),
                        page_number=page_number,
                    )
                )
                chunk_index += 1

            # Start new chunk with overlap from end of previous chunk
            overlap_text = _get_overlap(current_chunk, char_overlap)
            current_chunk = overlap_text + " " + para if overlap_text else para
        else:
            current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # Handle paragraphs that are themselves too long
        if len(current_chunk) > char_size * 2:
            sub_chunks = _force_split(current_chunk, char_size, char_overlap)
            for i, sub in enumerate(sub_chunks[:-1]):
                if len(sub) >= min_char_size:
                    chunks.append(
                        TextChunk(
                            content=sub.strip(),
                            chunk_index=chunk_index,
                            token_count=estimate_tokens(sub),
                            page_number=page_number,
                        )
                    )
                    chunk_index += 1
            current_chunk = sub_chunks[-1] if sub_chunks else ""

    # Flush remaining content
    if current_chunk and len(current_chunk) >= min_char_size:
        chunks.append(
            TextChunk(
                content=current_chunk.strip(),
                chunk_index=chunk_index,
                token_count=estimate_tokens(current_chunk),
                page_number=page_number,
            )
        )

    logger.debug(
        "text_chunked",
        input_length=len(text),
        chunk_count=len(chunks),
        chunk_size=chunk_size,
    )
    return chunks


def _split_into_paragraphs(text: str) -> List[str]:
    """Split text on double newlines (paragraph boundaries)."""
    import re
    # Split on double newlines, keeping single newlines within paragraphs
    paragraphs = re.split(r"\n\s*\n", text)
    return [p.strip() for p in paragraphs if p.strip()]


def _get_overlap(text: str, overlap_chars: int) -> str:
    """Get the last `overlap_chars` characters of text, aligned to word boundary."""
    if len(text) <= overlap_chars:
        return text
    overlap = text[-overlap_chars:]
    # Align to word boundary
    space_idx = overlap.find(" ")
    if space_idx > 0:
        overlap = overlap[space_idx + 1:]
    return overlap


def _force_split(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Force-split text that's too long, aligned to word boundaries."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to align to word boundary
            space_idx = text.rfind(" ", start, end)
            if space_idx > start:
                end = space_idx
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

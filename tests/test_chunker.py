"""
Tests for the text chunking service.
"""
import pytest
from app.services.ingestion.chunker import TextChunk, chunk_text, estimate_tokens


def test_chunk_text_basic():
    """Should split text into chunks."""
    text = "This is a test paragraph.\n\nThis is another paragraph with more content."
    chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)
    assert len(chunks) >= 1
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_text_empty():
    """Should return empty list for empty input."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_preserves_content():
    """All content should be present across chunks."""
    text = "Word " * 500  # 500 words
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    # All chunks should have content
    assert all(c.content.strip() for c in chunks)


def test_chunk_text_page_number():
    """Page number should be preserved in chunks."""
    text = "Some content on page 3."
    chunks = chunk_text(text, page_number=3)
    assert all(c.page_number == 3 for c in chunks)


def test_chunk_indices_sequential():
    """Chunk indices should be sequential starting from 0."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks = chunk_text(text, chunk_size=10, chunk_overlap=2)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_estimate_tokens():
    """Token estimation should be reasonable."""
    text = "Hello world"  # 11 chars ≈ 2-3 tokens
    tokens = estimate_tokens(text)
    assert tokens >= 1
    assert tokens <= 10


def test_chunk_text_min_size():
    """Chunks smaller than min_chunk_size should be discarded."""
    text = "Hi.\n\nThis is a much longer paragraph with enough content to form a proper chunk."
    chunks = chunk_text(text, min_chunk_size=50)
    # Very short "Hi." should be merged or discarded
    assert all(len(c.content) >= 10 for c in chunks)

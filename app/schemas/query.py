"""
Pydantic schemas for the Query API.
"""
import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request body for semantic search / RAG query."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="The search query or question",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
    )
    file_ids: Optional[List[uuid.UUID]] = Field(
        default=None,
        description="Restrict search to specific file IDs. None = search all files.",
    )
    use_llm: bool = Field(
        default=False,
        description="Whether to synthesize an answer using Gemini LLM",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Metadata filters (e.g., {'file_type': 'pdf'})",
    )
    min_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score threshold (0.0 = no filter)",
    )


class ChunkResult(BaseModel):
    """A single retrieved chunk with its similarity score."""

    chunk_id: uuid.UUID
    file_id: uuid.UUID
    filename: str
    file_type: Optional[str] = None
    content: str
    score: float = Field(..., description="Cosine similarity score (0-1)")
    chunk_index: int
    page_number: Optional[int] = None
    start_line: Optional[int] = None
    end_line: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}


class QueryResponse(BaseModel):
    """Response from the Query API."""

    query: str
    answer: Optional[str] = Field(
        None,
        description="LLM-synthesized answer (only present when use_llm=True)",
    )
    sources: List[ChunkResult] = Field(
        default_factory=list,
        description="Retrieved chunks ranked by relevance",
    )
    total_results: int
    cached: bool = Field(default=False, description="Whether this result was served from cache")
    model_used: Optional[str] = Field(None, description="LLM model used for synthesis")
    request_id: Optional[str] = None

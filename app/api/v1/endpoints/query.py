"""
Query API endpoint.
Handles semantic search and RAG-powered Q&A.
"""
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.db.session import get_db
from app.schemas.query import QueryRequest, QueryResponse
from app.services.query.rag_pipeline import RAGPipeline

logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Semantic search and RAG query",
    description=(
        "Search the knowledge base using semantic similarity. "
        "Optionally synthesize an answer using Gemini."
    ),
)
async def query_knowledge_base(
    request: Request,
    query_request: QueryRequest,
    db: AsyncSession = Depends(get_db),
    api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """
    Query the knowledge base with semantic search.

    - Embeds the query using Sentence Transformers (local)
    - Searches Qdrant Cloud for similar chunks
    - Optionally synthesizes an answer with Gemini
    - Returns ranked results with source citations
    """
    request_id = str(uuid.uuid4())

    logger.info(
        "query_request",
        query=query_request.query[:100],
        top_k=query_request.top_k,
        use_llm=query_request.use_llm,
        request_id=request_id,
    )

    try:
        pipeline = RAGPipeline(db=db)
        response = await pipeline.query(query_request, request_id=request_id)
        return response

    except Exception as e:
        logger.error("query_failed", error=str(e), request_id=request_id, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query processing failed: {str(e)}",
        )

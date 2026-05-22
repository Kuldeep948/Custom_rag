"""
RAG (Retrieval-Augmented Generation) query pipeline.
Handles semantic search and optional Gemini LLM synthesis.
No Redis dependency — stateless per request.
"""
import asyncio
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models.chunk import Chunk
from app.db.models.file import File
from app.schemas.query import ChunkResult, QueryRequest, QueryResponse
from app.services.embedding.embedder import get_embedder_singleton
from app.services.vector_store.qdrant_store import VectorSearchResult, get_vector_store

logger = get_logger(__name__)


class RAGPipeline:
    """
    Full RAG pipeline:
    1. Embed the query (Sentence Transformers, local)
    2. Search Qdrant Cloud for similar chunks
    3. Enrich results with PostgreSQL metadata
    4. (Optional) Synthesize answer with Gemini
    5. Return ranked results with citations
    """

    def __init__(self, db: AsyncSession):
        self.db = db
        self.embedder = get_embedder_singleton()
        self.vector_store = get_vector_store()

    async def query(self, request: QueryRequest, request_id: str = None) -> QueryResponse:
        """Execute the full RAG pipeline for a query."""
        logger.info(
            "rag_query_start",
            query=request.query[:100],
            top_k=request.top_k,
            use_llm=request.use_llm,
            request_id=request_id,
        )

        # 1. Embed the query
        query_embedding = await self.embedder.embed_query(request.query)

        # 2. Build metadata filter
        where_filter = self._build_where_filter(request)

        # 3. Vector similarity search
        vector_results = self.vector_store.search(
            query_embedding=query_embedding,
            top_k=request.top_k,
            where=where_filter,
            min_score=request.min_score,
        )

        if not vector_results:
            return QueryResponse(
                query=request.query,
                answer=None,
                sources=[],
                total_results=0,
                cached=False,
                request_id=request_id,
            )

        # 4. Enrich with PostgreSQL metadata
        chunk_results = await self._enrich_results(vector_results)

        # 5. Optional Gemini synthesis
        answer = None
        model_used = None
        if request.use_llm and chunk_results:
            answer, model_used = await self._synthesize_with_gemini(
                query=request.query,
                chunks=chunk_results,
            )

        logger.info(
            "rag_query_complete",
            results=len(chunk_results),
            has_answer=answer is not None,
            request_id=request_id,
        )

        return QueryResponse(
            query=request.query,
            answer=answer,
            sources=chunk_results,
            total_results=len(chunk_results),
            cached=False,
            model_used=model_used,
            request_id=request_id,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    async def _enrich_results(
        self, vector_results: List[VectorSearchResult]
    ) -> List[ChunkResult]:
        """Fetch chunk + file metadata from PostgreSQL to enrich vector results."""
        chunk_ids = [uuid.UUID(r.id) for r in vector_results]

        stmt = (
            select(Chunk, File)
            .join(File, Chunk.file_id == File.id)
            .where(
                Chunk.id.in_(chunk_ids),
                Chunk.deleted_at.is_(None),
                File.deleted_at.is_(None),
            )
        )
        result = await self.db.execute(stmt)
        rows = result.all()

        chunk_map: Dict[str, tuple] = {
            str(chunk.id): (chunk, file) for chunk, file in rows
        }

        enriched = []
        for vr in vector_results:
            if vr.id not in chunk_map:
                continue
            chunk, file = chunk_map[vr.id]
            enriched.append(
                ChunkResult(
                    chunk_id=chunk.id,
                    file_id=file.id,
                    filename=file.original_filename,
                    file_type=file.file_type,
                    content=chunk.content,
                    score=round(vr.score, 4),
                    chunk_index=chunk.chunk_index,
                    page_number=chunk.page_number,
                    start_line=chunk.start_line,
                    end_line=chunk.end_line,
                    function_name=chunk.function_name,
                    class_name=chunk.class_name,
                    metadata=chunk.metadata_,
                )
            )
        return enriched

    async def _synthesize_with_gemini(
        self,
        query: str,
        chunks: List[ChunkResult],
    ) -> tuple[Optional[str], Optional[str]]:
        """Synthesize a grounded answer using Google Gemini."""
        if not settings.gemini_api_key:
            logger.warning("gemini_synthesis_skipped", reason="no_gemini_api_key")
            return None, None

        try:
            import google.generativeai as genai

            genai.configure(api_key=settings.gemini_api_key)

            # Build numbered context block
            context_parts = []
            for i, chunk in enumerate(chunks[:5], 1):
                src = f"[Source {i}: {chunk.filename}"
                if chunk.page_number:
                    src += f", page {chunk.page_number}"
                if chunk.function_name:
                    src += f", fn '{chunk.function_name}'"
                src += "]"
                context_parts.append(f"{src}\n{chunk.content}")
            context = "\n\n---\n\n".join(context_parts)

            user_prompt = (
                f"Context:\n{context}\n\n"
                f"Question: {query}\n\n"
                f"Answer based on the context above:"
            )

            model = genai.GenerativeModel(
                model_name=settings.llm_model,
                system_instruction=(
                    "You are a helpful assistant that answers questions based strictly on "
                    "the provided context. If the answer is not found in the context, "
                    "say so clearly. Always cite source numbers when referencing information."
                ),
                generation_config=genai.GenerationConfig(
                    max_output_tokens=settings.llm_max_tokens,
                    temperature=settings.llm_temperature,
                ),
            )

            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, lambda: model.generate_content(user_prompt)
            )

            logger.info("gemini_synthesis_complete", model=settings.llm_model)
            return response.text, settings.llm_model

        except Exception as e:
            logger.error("gemini_synthesis_failed", error=str(e))
            return None, None

    def _build_where_filter(
        self, request: QueryRequest
    ) -> Optional[Dict[str, Any]]:
        """Build Qdrant metadata filter from request parameters."""
        conditions = []

        if request.file_ids:
            file_id_strs = [str(fid) for fid in request.file_ids]
            if len(file_id_strs) == 1:
                conditions.append({"file_id": {"$eq": file_id_strs[0]}})
            else:
                conditions.append({"file_id": {"$in": file_id_strs}})

        if request.filters:
            for key, value in request.filters.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append({key: {"$eq": value}})

        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}

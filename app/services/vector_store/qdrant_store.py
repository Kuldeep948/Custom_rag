"""
Qdrant vector store service.

Supports both:
  - Local mode  : qdrant running as a Docker container (QDRANT_URL)
  - Cloud mode  : Qdrant Cloud cluster (QDRANT_URL + QDRANT_API_KEY)
"""
from typing import Any, Dict, List, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Map human-readable distance names → Qdrant Distance enum
_DISTANCE_MAP = {
    "cosine": qmodels.Distance.COSINE,
    "dot":    qmodels.Distance.DOT,
    "euclid": qmodels.Distance.EUCLID,
    "l2":     qmodels.Distance.EUCLID,
}


class VectorSearchResult:
    """A single result from a vector similarity search."""

    def __init__(self, id: str, content: str, score: float, metadata: Dict[str, Any]):
        self.id = id
        self.content = content
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:
        return f"<VectorSearchResult id={self.id} score={self.score:.4f}>"


class QdrantVectorStore:
    """
    Qdrant-backed vector store.

    All public methods:
    add_chunks, search, delete_by_file, delete_chunks,
    get_collection_stats, health_check
    """

    def __init__(self) -> None:
        self._client: Optional[QdrantClient] = None

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            kwargs: Dict[str, Any] = {
                "url": settings.qdrant_url,
                "check_compatibility": False,   # we verify compatibility manually
            }
            if settings.qdrant_api_key:
                kwargs["api_key"] = settings.qdrant_api_key
            if settings.qdrant_grpc_port:
                kwargs["grpc_port"] = settings.qdrant_grpc_port
                kwargs["prefer_grpc"] = True

            self._client = QdrantClient(**kwargs)
            logger.info("qdrant_client_initialized", url=settings.qdrant_url)
        return self._client

    def _ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        client = self._get_client()
        collection_name = settings.qdrant_collection_name
        distance = _DISTANCE_MAP.get(
            settings.qdrant_distance_metric.lower(), qmodels.Distance.COSINE
        )

        try:
            client.get_collection(collection_name)
            logger.debug("qdrant_collection_exists", collection=collection_name)
        except (UnexpectedResponse, Exception):
            # Collection does not exist — create it
            client.create_collection(
                collection_name=collection_name,
                vectors_config=qmodels.VectorParams(
                    size=settings.embedding_dimension,
                    distance=distance,
                    on_disk=settings.qdrant_on_disk_vectors,
                ),
                # HNSW index tuning (good defaults for RAG workloads)
                hnsw_config=qmodels.HnswConfigDiff(
                    m=16,
                    ef_construct=100,
                    full_scan_threshold=10_000,
                ),
                optimizers_config=qmodels.OptimizersConfigDiff(
                    indexing_threshold=20_000,
                ),
            )
            logger.info(
                "qdrant_collection_created",
                collection=collection_name,
                dimension=settings.embedding_dimension,
                distance=distance,
            )

    # ── Public API ────────────────────────────────────────────────────────────

    def add_chunks(
        self,
        chunk_ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """
        Upsert chunks into Qdrant.

        Args:
            chunk_ids  : UUID strings — used as Qdrant point IDs
            embeddings : Dense vectors (must match collection dimension)
            documents  : Raw text stored in the payload for retrieval
            metadatas  : Arbitrary key-value payload for filtering
        """
        if not chunk_ids:
            return

        self._ensure_collection()
        client = self._get_client()

        points = [
            qmodels.PointStruct(
                id=chunk_id,
                vector=embedding,
                payload={
                    "content": document,
                    **self._sanitize_payload(metadata),
                },
            )
            for chunk_id, embedding, document, metadata in zip(
                chunk_ids, embeddings, documents, metadatas
            )
        ]

        # Upsert in batches of 256 to stay within Qdrant's recommended limits
        batch_size = 256
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            client.upsert(
                collection_name=settings.qdrant_collection_name,
                points=batch,
                wait=True,  # wait for indexing to complete
            )

        logger.info(
            "chunks_upserted_to_qdrant",
            count=len(points),
            collection=settings.qdrant_collection_name,
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
        min_score: float = 0.0,
    ) -> List[VectorSearchResult]:
        """
        Semantic similarity search in Qdrant.

        Args:
            query_embedding : Dense query vector
            top_k           : Maximum number of results
            where           : Filter dict — supports two formats:
                              • Simple equality  : {"file_id": "uuid-string"}
                              • Multi-condition  : {"$and": [{"file_id": ...}, ...]}
            min_score       : Discard results below this score threshold

        Returns:
            List[VectorSearchResult] sorted by score descending
        """
        self._ensure_collection()
        client = self._get_client()

        qdrant_filter = self._build_qdrant_filter(where) if where else None

        try:
            hits = client.search(
                collection_name=settings.qdrant_collection_name,
                query_vector=query_embedding,
                limit=top_k,
                query_filter=qdrant_filter,
                score_threshold=min_score if min_score > 0.0 else None,
                with_payload=True,
            )
        except Exception as e:
            logger.error("qdrant_search_failed", error=str(e))
            return []

        results = []
        for hit in hits:
            payload = hit.payload or {}
            content = payload.pop("content", "")
            results.append(
                VectorSearchResult(
                    id=str(hit.id),
                    content=content,
                    score=round(hit.score, 6),
                    metadata=payload,
                )
            )

        logger.debug(
            "qdrant_search_complete",
            results=len(results),
            top_k=top_k,
            collection=settings.qdrant_collection_name,
        )
        return results

    def delete_by_file(self, file_id: str) -> int:
        """
        Delete all points that belong to a given file_id.

        Returns the number of points deleted.
        """
        self._ensure_collection()
        client = self._get_client()

        # First count how many points match so we can return the count
        count_result = client.count(
            collection_name=settings.qdrant_collection_name,
            count_filter=qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="file_id",
                        match=qmodels.MatchValue(value=file_id),
                    )
                ]
            ),
            exact=True,
        )
        deleted_count = count_result.count

        if deleted_count == 0:
            return 0

        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="file_id",
                            match=qmodels.MatchValue(value=file_id),
                        )
                    ]
                )
            ),
            wait=True,
        )

        logger.info(
            "qdrant_points_deleted_by_file",
            file_id=file_id,
            count=deleted_count,
        )
        return deleted_count

    def delete_chunks(self, chunk_ids: List[str]) -> None:
        """Delete specific points by their UUID string IDs."""
        if not chunk_ids:
            return
        self._ensure_collection()
        client = self._get_client()
        client.delete(
            collection_name=settings.qdrant_collection_name,
            points_selector=qmodels.PointIdsList(points=chunk_ids),
            wait=True,
        )
        logger.info("qdrant_chunks_deleted", count=len(chunk_ids))

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return collection info for the metrics endpoint."""
        try:
            client = self._get_client()
            info = client.get_collection(settings.qdrant_collection_name)
            return {
                "provider": "qdrant",
                "collection_name": settings.qdrant_collection_name,
                "total_vectors": info.vectors_count or 0,
                "indexed_vectors": info.indexed_vectors_count or 0,
                "status": str(info.status),
                "qdrant_url": settings.qdrant_url,
            }
        except Exception as e:
            logger.error("qdrant_stats_failed", error=str(e))
            return {"provider": "qdrant", "error": str(e)}

    def health_check(self) -> bool:
        """Return True if Qdrant is reachable and the collection is accessible."""
        try:
            client = self._get_client()
            client.get_collections()   # lightweight ping
            return True
        except Exception as e:
            logger.error("qdrant_health_check_failed", error=str(e))
            return False

    # ── Private helpers ───────────────────────────────────────────────────────

    def _sanitize_payload(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Qdrant payload values can be str / int / float / bool / list / dict.
        Convert anything else to string.
        """
        sanitized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool, list, dict)):
                sanitized[key] = value
            elif value is None:
                sanitized[key] = ""
            else:
                sanitized[key] = str(value)
        return sanitized

    def _build_qdrant_filter(self, where: Dict[str, Any]) -> qmodels.Filter:
        """
        Convert the generic filter dict (same format used by the RAG pipeline)
        into a Qdrant Filter object.

        Supported formats
        -----------------
        Simple equality:
            {"file_id": "some-uuid"}
            {"file_id": {"$eq": "some-uuid"}}
            {"file_id": {"$in": ["uuid1", "uuid2"]}}

        AND of conditions:
            {"$and": [{"file_id": "uuid"}, {"file_type": "pdf"}]}
        """
        must_conditions: List[qmodels.Condition] = []

        if "$and" in where:
            for sub in where["$and"]:
                must_conditions.extend(self._parse_conditions(sub))
        else:
            must_conditions.extend(self._parse_conditions(where))

        return qmodels.Filter(must=must_conditions)

    def _parse_conditions(
        self, condition: Dict[str, Any]
    ) -> List[qmodels.Condition]:
        """Parse a single condition dict into a list of Qdrant conditions."""
        result: List[qmodels.Condition] = []
        for key, value in condition.items():
            if key.startswith("$"):
                continue  # top-level operators handled by caller

            if isinstance(value, dict):
                op = list(value.keys())[0]
                val = list(value.values())[0]
                if op == "$eq":
                    result.append(
                        qmodels.FieldCondition(
                            key=key, match=qmodels.MatchValue(value=val)
                        )
                    )
                elif op == "$in":
                    result.append(
                        qmodels.FieldCondition(
                            key=key, match=qmodels.MatchAny(any=val)
                        )
                    )
            else:
                # Plain equality shorthand: {"file_id": "uuid"}
                result.append(
                    qmodels.FieldCondition(
                        key=key, match=qmodels.MatchValue(value=value)
                    )
                )
        return result


# ── Module-level singleton ────────────────────────────────────────────────────
_vector_store_instance: Optional[QdrantVectorStore] = None


def get_vector_store() -> QdrantVectorStore:
    """Return the global QdrantVectorStore singleton."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = QdrantVectorStore()
    return _vector_store_instance

"""
Tests for the Query API endpoint.
"""
import pytest
from httpx import AsyncClient
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.mark.asyncio
async def test_query_requires_auth(client: AsyncClient):
    """Query endpoint should require API key."""
    from httpx import AsyncClient as RawClient
    from httpx import ASGITransport
    from app.main import app

    async with RawClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as unauthenticated:
        response = await unauthenticated.post(
            "/api/v1/query",
            json={"query": "test query"},
        )
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_query_empty_knowledge_base(client: AsyncClient):
    """Query against empty knowledge base should return empty results."""
    with patch("app.services.embedding.embedder.get_embedder_singleton") as mock_embedder:
        mock_emb = AsyncMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 1536)
        mock_embedder.return_value = mock_emb

        with patch("app.services.vector_store.qdrant_store.get_vector_store") as mock_vs:
            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_vs.return_value = mock_store

            response = await client.post(
                "/api/v1/query",
                json={"query": "What is the system design?", "top_k": 5},
            )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What is the system design?"
    assert data["sources"] == []
    assert data["total_results"] == 0


@pytest.mark.asyncio
async def test_query_validation(client: AsyncClient):
    """Query should validate request parameters."""
    # Empty query
    response = await client.post(
        "/api/v1/query",
        json={"query": ""},
    )
    assert response.status_code == 422

    # top_k out of range
    response = await client.post(
        "/api/v1/query",
        json={"query": "test", "top_k": 100},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_response_structure(client: AsyncClient):
    """Query response should have the correct structure."""
    with patch("app.services.embedding.embedder.get_embedder_singleton") as mock_embedder:
        mock_emb = AsyncMock()
        mock_emb.embed_query = AsyncMock(return_value=[0.1] * 384)
        mock_embedder.return_value = mock_emb

        with patch("app.services.vector_store.qdrant_store.get_vector_store") as mock_vs:
            mock_store = MagicMock()
            mock_store.search.return_value = []
            mock_vs.return_value = mock_store

            response = await client.post(
                "/api/v1/query",
                json={
                    "query": "test query",
                    "top_k": 3,
                    "use_llm": False,
                },
            )

    assert response.status_code == 200
    data = response.json()

    # Verify response structure
    assert "query" in data
    assert "sources" in data
    assert "total_results" in data
    assert "cached" in data
    assert isinstance(data["sources"], list)
    assert isinstance(data["total_results"], int)

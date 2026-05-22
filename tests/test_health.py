"""
Tests for health and metrics endpoints.
"""
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient):
    """Health endpoint should return 200 with service statuses."""
    response = await client.get("/api/v1/health")
    assert response.status_code == 200

    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "services" in data
    assert data["status"] in ("healthy", "degraded", "unhealthy")


@pytest.mark.asyncio
async def test_root_endpoint(client: AsyncClient):
    """Root endpoint should return app info."""
    response = await client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "name" in data
    assert "version" in data


@pytest.mark.asyncio
async def test_metrics_requires_auth(client: AsyncClient):
    """Metrics endpoint requires API key."""
    from httpx import AsyncClient as RawClient
    from httpx import ASGITransport
    from app.main import app

    async with RawClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as unauthenticated:
        response = await unauthenticated.get("/api/v1/metrics")
        assert response.status_code == 401

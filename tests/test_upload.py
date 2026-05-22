"""
Tests for the Upload API endpoint.
"""
import io
import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_upload_python_file(client: AsyncClient, sample_python_content: bytes):
    """Should successfully upload a Python file."""
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("test_code.py", io.BytesIO(sample_python_content), "text/x-python")},
    )
    assert response.status_code == 202

    data = response.json()
    assert "file_id" in data
    assert "job_id" in data
    assert data["filename"] == "test_code.py"
    assert data["file_type"] == "python"
    assert data["status"] == "pending"


@pytest.mark.asyncio
async def test_upload_requires_auth(client: AsyncClient, sample_python_content: bytes):
    """Upload should reject requests without API key."""
    from httpx import AsyncClient as RawClient
    from httpx import ASGITransport
    from app.main import app

    async with RawClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as unauthenticated:
        response = await unauthenticated.post(
            "/api/v1/upload",
            files={"file": ("test.py", io.BytesIO(sample_python_content), "text/x-python")},
        )
        assert response.status_code == 401


@pytest.mark.asyncio
async def test_upload_unsupported_file_type(client: AsyncClient):
    """Should reject unsupported file types."""
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("test.exe", io.BytesIO(b"binary content"), "application/octet-stream")},
    )
    assert response.status_code == 415


@pytest.mark.asyncio
async def test_upload_empty_file(client: AsyncClient):
    """Should reject empty files."""
    response = await client.post(
        "/api/v1/upload",
        files={"file": ("empty.py", io.BytesIO(b""), "text/x-python")},
    )
    assert response.status_code == 400


@pytest.mark.asyncio
async def test_upload_with_metadata(client: AsyncClient, sample_python_content: bytes):
    """Should accept and store metadata with the file."""
    import json
    metadata = json.dumps({"source": "test", "tags": ["python", "sample"]})

    response = await client.post(
        "/api/v1/upload",
        files={"file": ("meta_test.py", io.BytesIO(sample_python_content), "text/x-python")},
        data={"metadata": metadata},
    )
    assert response.status_code == 202
    data = response.json()
    assert "file_id" in data


@pytest.mark.asyncio
async def test_upload_duplicate_detection(client: AsyncClient, sample_python_content: bytes):
    """Should detect and reject duplicate files."""
    # First upload
    response1 = await client.post(
        "/api/v1/upload",
        files={"file": ("dup_test.py", io.BytesIO(sample_python_content), "text/x-python")},
    )
    assert response1.status_code == 202

    # Second upload of same content
    response2 = await client.post(
        "/api/v1/upload",
        files={"file": ("dup_test_copy.py", io.BytesIO(sample_python_content), "text/x-python")},
    )
    assert response2.status_code == 409
    assert "already been ingested" in response2.json()["detail"]["message"]

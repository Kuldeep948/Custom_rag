"""
Health check and metrics endpoints.
"""
import time

from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.logging import get_logger
from app.core.security import verify_api_key
from app.db.session import get_db
from app.schemas.health import HealthResponse, MetricsResponse, ServiceStatus
from app.services.vector_store.qdrant_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health check",
)
async def health_check(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> HealthResponse:
    """Check the health of all system components."""
    services = {}
    overall_status = "healthy"

    # ── PostgreSQL ────────────────────────────────────────────────
    try:
        start = time.monotonic()
        await db.execute(text("SELECT 1"))
        latency = (time.monotonic() - start) * 1000
        services["postgresql"] = ServiceStatus(
            status="healthy", latency_ms=round(latency, 2)
        )
    except Exception as e:
        services["postgresql"] = ServiceStatus(status="unhealthy", details=str(e))
        overall_status = "unhealthy"

    # ── Qdrant ────────────────────────────────────────────────────
    try:
        start = time.monotonic()
        is_healthy = get_vector_store().health_check()
        latency = (time.monotonic() - start) * 1000
        services["qdrant"] = ServiceStatus(
            status="healthy" if is_healthy else "unhealthy",
            latency_ms=round(latency, 2),
        )
        if not is_healthy:
            overall_status = "unhealthy"
    except Exception as e:
        services["qdrant"] = ServiceStatus(status="unhealthy", details=str(e))
        overall_status = "unhealthy"

    return HealthResponse(
        status=overall_status,
        version=settings.app_version,
        environment=settings.environment,
        services=services,
    )


@router.get(
    "/metrics",
    response_model=MetricsResponse,
    summary="Processing statistics",
    dependencies=[Depends(verify_api_key)],
)
async def get_metrics(
    db: AsyncSession = Depends(get_db),
) -> MetricsResponse:
    """Return processing statistics and system metrics."""
    from sqlalchemy import func, select
    from app.db.models.file import File
    from app.db.models.chunk import Chunk

    total_files = (
        await db.execute(
            select(func.count(File.id)).where(File.deleted_at.is_(None))
        )
    ).scalar_one()

    total_chunks = (
        await db.execute(
            select(func.count(Chunk.id)).where(Chunk.deleted_at.is_(None))
        )
    ).scalar_one()

    status_rows = (
        await db.execute(
            select(File.status, func.count(File.id))
            .where(File.deleted_at.is_(None))
            .group_by(File.status)
        )
    ).all()
    files_by_status = {row[0]: row[1] for row in status_rows}

    type_rows = (
        await db.execute(
            select(File.file_type, func.count(File.id))
            .where(File.deleted_at.is_(None), File.file_type.isnot(None))
            .group_by(File.file_type)
        )
    ).all()
    files_by_type = {row[0]: row[1] for row in type_rows}

    vector_store_size = None
    try:
        stats = get_vector_store().get_collection_stats()
        vector_store_size = stats.get("total_vectors")
    except Exception:
        pass

    return MetricsResponse(
        total_files=total_files,
        total_chunks=total_chunks,
        files_by_status=files_by_status,
        files_by_type=files_by_type,
        total_queries=0,
        cache_hit_rate=None,
        vector_store_size=vector_store_size,
    )

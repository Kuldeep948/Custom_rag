"""
Pydantic schemas for health and metrics endpoints.
"""
from typing import Any, Dict, Optional

from pydantic import BaseModel


class ServiceStatus(BaseModel):
    status: str  # healthy | degraded | unhealthy
    latency_ms: Optional[float] = None
    details: Optional[str] = None


class HealthResponse(BaseModel):
    status: str  # healthy | degraded | unhealthy
    version: str
    environment: str
    services: Dict[str, ServiceStatus]


class MetricsResponse(BaseModel):
    total_files: int
    total_chunks: int
    files_by_status: Dict[str, int]
    files_by_type: Dict[str, int]
    total_queries: int
    avg_query_latency_ms: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    vector_store_size: Optional[int] = None

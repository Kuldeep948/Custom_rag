"""
API v1 router — aggregates all endpoint routers.
"""
from fastapi import APIRouter

from app.api.v1.endpoints import files, health, query, upload

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(upload.router, tags=["Ingestion"])
api_router.include_router(query.router, tags=["Query"])
api_router.include_router(files.router, tags=["File Management"])
api_router.include_router(health.router, tags=["System"])

# fitz_sage/api/routes/health.py
"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool

from fitz_sage.api.dependencies import get_fitz_version, get_service
from fitz_sage.api.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns server status, version, and component health.
    """
    service = get_service()
    health_result = await run_in_threadpool(service.health_check)

    return HealthResponse(
        status="healthy" if health_result.healthy else "unhealthy",
        version=get_fitz_version(),
        components=health_result.components,
    )

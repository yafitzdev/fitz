# fitz_ai/api/routes/health.py
"""Health check endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from fitz_ai.api.dependencies import config_exists, get_fitz_version
from fitz_ai.api.models.schemas import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """
    Health check endpoint.

    Returns server status, version, and configuration state.
    """
    return HealthResponse(
        status="healthy",
        version=get_fitz_version(),
        config_exists=config_exists(),
    )

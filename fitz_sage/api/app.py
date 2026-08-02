# fitz_sage/api/app.py
"""FastAPI application for the fitz-sage REST API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fitz_sage.api.dependencies import get_fitz_version
from fitz_sage.api.routes import (
    collections_router,
    health_router,
    query_router,
)
from fitz_sage.api.security import LocalOrApiKeyMiddleware, allowed_api_origins


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="fitz-sage API",
        description="REST API for fitz-sage. Query knowledge bases and manage collections.",
        version=get_fitz_version(),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    origins = allowed_api_origins()
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["Content-Type", "X-Fitz-API-Key"],
        )
    app.add_middleware(LocalOrApiKeyMiddleware)

    # Register routes
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(collections_router)

    return app


# Default app instance for uvicorn
app = create_app()

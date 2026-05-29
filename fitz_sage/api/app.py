# fitz_sage/api/app.py
"""FastAPI application for Fitz RAG API."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from fitz_sage.api.dependencies import get_fitz_version
from fitz_sage.api.routes import (
    collections_router,
    health_router,
    query_router,
)


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI app instance.
    """
    app = FastAPI(
        title="Fitz RAG API",
        description=(
            "REST API for the Fitz RAG framework. " "Query knowledge bases and manage collections."
        ),
        version=get_fitz_version(),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    # Add CORS middleware for browser clients. allow_credentials must be False
    # when origins is "*" — browsers reject the wildcard+credentials combination,
    # and this API is unauthenticated (no cookies/credentials to send).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(query_router)
    app.include_router(collections_router)

    return app


# Default app instance for uvicorn
app = create_app()

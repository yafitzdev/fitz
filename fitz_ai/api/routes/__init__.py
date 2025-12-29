# fitz_ai/api/routes/__init__.py
"""API route modules."""

from fitz_ai.api.routes.collections import router as collections_router
from fitz_ai.api.routes.health import router as health_router
from fitz_ai.api.routes.ingest import router as ingest_router
from fitz_ai.api.routes.query import router as query_router

__all__ = [
    "collections_router",
    "health_router",
    "ingest_router",
    "query_router",
]

# fitz_sage/api/__init__.py
"""
Fitz REST API.

Provides HTTP endpoints for the Fitz RAG framework.
"""

from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any) -> Any:
    """Create the FastAPI app, importing optional API dependencies lazily."""
    from fitz_sage.api.app import create_app as _create_app

    return _create_app(*args, **kwargs)


__all__ = ["create_app"]

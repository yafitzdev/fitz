# core/vector_db/__init__.py
from __future__ import annotations

from fitz.vector_db.base import SearchResult, VectorDBPlugin

__all__ = ["SearchResult", "VectorDBEngine", "VectorDBPlugin"]

# fitz/vector_db/__init__.py

from __future__ import annotations

from fitz_ai.vector_db.base import SearchResult, VectorDBPlugin
from fitz_ai.vector_db.loader import create_vector_db_plugin, load_vector_db_spec

__all__ = [
    "SearchResult",
    "VectorDBPlugin",
    "create_vector_db_plugin",
    "load_vector_db_spec",
]

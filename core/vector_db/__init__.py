"""
Vector DB subsystem for fitz-stack.

This package exposes:

- VectorRecord, SearchResult: simple data containers
- VectorDBPlugin: protocol for all vector DB implementations
- get_vector_db_plugin / register_vector_db_plugin: registry helpers
"""

from .base import VectorDBPlugin, VectorRecord, SearchResult
from .registry import (
    get_vector_db_plugin,
    register_vector_db_plugin,
    VectorDBRegistryError,
)

__all__ = [
    "VectorDBPlugin",
    "VectorRecord",
    "SearchResult",
    "get_vector_db_plugin",
    "register_vector_db_plugin",
    "VectorDBRegistryError",
]

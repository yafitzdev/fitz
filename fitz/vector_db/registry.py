# fitz/vector_db/registry.py
"""
Vector DB plugin registry.

Handles discovery and registration of vector database plugins.
Separate from LLM registry for cleaner architecture.

Design principle: NO SILENT FALLBACK
- If user configures "qdrant", they get qdrant or an error
- If user wants local-faiss, they explicitly configure "local-faiss"
- No magic substitution that could cause confusion
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from fitz.core.registry import (  # Functions; Registry; Errors
    VECTOR_DB_REGISTRY,
    PluginNotFoundError,
    PluginRegistryError,
    VectorDBRegistryError,
    available_vector_db_plugins,
    get_vector_db_plugin,
    resolve_vector_db_plugin,
)

if TYPE_CHECKING:
    pass

__all__ = [
    # Functions
    "get_vector_db_plugin",
    "available_vector_db_plugins",
    "resolve_vector_db_plugin",
    # Registry
    "VECTOR_DB_REGISTRY",
    # Errors
    "VectorDBRegistryError",
    "PluginRegistryError",
    "PluginNotFoundError",
]

# fitz_ai/api/dependencies.py
"""Shared dependencies for API routes."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from fitz_ai.core.config import load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.sdk import fitz
from fitz_ai.vector_db.registry import get_vector_db_plugin


@lru_cache(maxsize=16)
def get_fitz_instance(collection: str = "default") -> fitz:
    """
    Get a cached fitz instance for the specified collection.

    Instances are cached per collection to avoid re-initialization overhead.
    """
    return fitz(collection=collection)


def get_vector_db() -> Any:
    """
    Get the configured vector DB plugin instance.

    Returns:
        Vector DB plugin with list_collections, get_collection_stats, etc.
    """
    config_path = FitzPaths.config()

    if not config_path.exists():
        # Return local FAISS with defaults if no config
        return get_vector_db_plugin("local_faiss")

    config = load_config_dict(config_path)
    vdb_config = config.get("vector_db", {})

    return get_vector_db_plugin(
        vdb_config.get("plugin_name", "local_faiss"),
        **vdb_config.get("kwargs", {}),
    )


def config_exists() -> bool:
    """Check if the fitz config file exists."""
    return FitzPaths.config().exists()


def get_fitz_version() -> str:
    """Get the current fitz version."""
    try:
        from fitz_ai import __version__

        return __version__
    except ImportError:
        return "unknown"

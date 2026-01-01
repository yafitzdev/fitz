# fitz_ai/cli/utils.py
"""
Shared CLI utilities.

Common functions used across multiple CLI commands.
Prefer using CLIContext directly for new code.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from fitz_ai.cli.context import CLIContext
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def load_fitz_rag_config() -> Tuple[Optional[dict], Optional[Any]]:
    """
    Load fitz_rag config.

    Returns:
        Tuple of (raw_config_dict, typed_config) or (None, None) if config not found.

    Note:
        Prefer using CLIContext.load() or CLIContext.load_or_none() directly.
    """
    ctx = CLIContext.load_or_none()
    if ctx is None:
        return None, None
    return ctx.raw_config, ctx.typed_config


def get_collections(config: dict) -> List[str]:
    """
    Get list of collections from vector DB.

    Args:
        config: Raw config dictionary containing vector_db settings.

    Returns:
        Sorted list of collection names, or empty list on error.

    Note:
        Prefer using ctx.get_collections() from CLIContext.
    """
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    try:
        vdb_plugin = config.get("vector_db", {}).get("plugin_name", "local_faiss")
        vdb_kwargs = config.get("vector_db", {}).get("kwargs", {})
        vdb = get_vector_db_plugin(vdb_plugin, **vdb_kwargs)
        return sorted(vdb.list_collections())
    except Exception:
        return []


def get_vector_db_client(config: dict):
    """
    Get vector DB client from config.

    Args:
        config: Raw config dictionary containing vector_db settings.

    Returns:
        Vector DB client instance.

    Note:
        Prefer using ctx.get_vector_db_client() from CLIContext.
    """
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    vdb_plugin = config.get("vector_db", {}).get("plugin_name", "local_faiss")
    vdb_kwargs = config.get("vector_db", {}).get("kwargs", {})
    return get_vector_db_plugin(vdb_plugin, **vdb_kwargs)


__all__ = [
    "CLIContext",
    "load_fitz_rag_config",
    "get_collections",
    "get_vector_db_client",
]

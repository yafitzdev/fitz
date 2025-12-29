# fitz_ai/cli/utils.py
"""
Shared CLI utilities.

Common functions used across multiple CLI commands.
"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


def load_classic_rag_config() -> Tuple[Optional[dict], Optional[Any]]:
    """
    Load classic_rag config.

    Returns:
        Tuple of (raw_config_dict, typed_config) or (None, None) if config not found.
    """
    try:
        from fitz_ai.engines.classic_rag.config import load_config

        config_path = FitzPaths.config()
        raw_config = load_config_dict(config_path)
        typed_config = load_config(config_path)
        return raw_config, typed_config
    except ConfigNotFoundError:
        return None, None
    except Exception:
        return None, None


def get_collections(config: dict) -> List[str]:
    """
    Get list of collections from vector DB.

    Args:
        config: Raw config dictionary containing vector_db settings.

    Returns:
        Sorted list of collection names, or empty list on error.
    """
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    try:
        vdb_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
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
    """
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    vdb_plugin = config.get("vector_db", {}).get("plugin_name", "local-faiss")
    vdb_kwargs = config.get("vector_db", {}).get("kwargs", {})
    return get_vector_db_plugin(vdb_plugin, **vdb_kwargs)


__all__ = [
    "load_classic_rag_config",
    "get_collections",
    "get_vector_db_client",
]

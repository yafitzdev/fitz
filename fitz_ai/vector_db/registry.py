# fitz_ai/vector_db/registry.py
"""
Vector DB plugin registry - YAML-based system.

Handles discovery and instantiation of vector DB plugins from YAML specs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

from fitz_ai.vector_db.loader import create_vector_db_plugin


def get_vector_db_plugin(plugin_name: str, **kwargs: Any) -> Any:
    """
    Get a vector DB plugin instance.

    Args:
        plugin_name: Name of the plugin (e.g., 'qdrant', 'pinecone', 'local-faiss')
        **kwargs: Plugin configuration (host, port, etc.)

    Returns:
        Vector DB plugin instance
    """
    return create_vector_db_plugin(plugin_name, **kwargs)


def available_vector_db_plugins() -> List[str]:
    """
    List available vector DB plugins.

    Returns:
        Sorted list of plugin names
    """
    plugins_dir = Path(__file__).parent / "plugins"

    if not plugins_dir.exists():
        return []

    return sorted(f.stem for f in plugins_dir.glob("*.yaml"))


__all__ = [
    "get_vector_db_plugin",
    "available_vector_db_plugins",
]

# fitz_ai/ingest/chunking/registry.py
"""Chunking plugin registry."""

from __future__ import annotations

from typing import Any, Dict, List, Type

from fitz_ai.core.registry import CHUNKING_REGISTRY, PluginNotFoundError


def get_chunking_plugin(plugin_name: str) -> Type[Any]:
    """Get a chunking plugin by name."""
    return CHUNKING_REGISTRY.get(plugin_name)


def available_chunking_plugins() -> List[str]:
    """List available chunking plugins."""
    return CHUNKING_REGISTRY.list_available()


def get_extension_to_chunker_map() -> Dict[str, str]:
    """
    Build a map of file extensions to chunker plugin names.

    Scans all registered chunking plugins and builds a mapping based on
    each plugin's `supported_extensions` attribute.

    Returns:
        Dict mapping extensions (e.g., ".md") to plugin names (e.g., "markdown").
    """
    CHUNKING_REGISTRY._ensure_discovered()

    ext_map: Dict[str, str] = {}
    for plugin_name, plugin_cls in CHUNKING_REGISTRY._plugins.items():
        # Get supported_extensions from the class
        # It might be a field with default_factory, so we need to handle both cases
        extensions = getattr(plugin_cls, "supported_extensions", None)

        # If it's a dataclass field with default_factory, we need to instantiate
        # to get the actual list. For efficiency, check the __dataclass_fields__.
        if extensions is None:
            fields = getattr(plugin_cls, "__dataclass_fields__", {})
            if "supported_extensions" in fields:
                field_info = fields["supported_extensions"]
                if field_info.default_factory is not None:
                    extensions = field_info.default_factory()

        if extensions:
            for ext in extensions:
                # Normalize to lowercase with dot
                ext_lower = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
                if ext_lower not in ext_map:
                    ext_map[ext_lower] = plugin_name

    return ext_map


def get_chunker_for_extension(ext: str) -> str | None:
    """
    Get the best chunker plugin name for a file extension.

    Args:
        ext: File extension (e.g., ".md", "py", ".PDF")

    Returns:
        Plugin name if a specialized chunker exists, None otherwise.
    """
    ext_lower = ext.lower() if ext.startswith(".") else f".{ext.lower()}"
    ext_map = get_extension_to_chunker_map()
    return ext_map.get(ext_lower)


__all__ = [
    "get_chunking_plugin",
    "available_chunking_plugins",
    "get_extension_to_chunker_map",
    "get_chunker_for_extension",
    "PluginNotFoundError",
]

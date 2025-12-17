# fitz/llm/registry.py
"""
Central LLM plugin registry.

Handles auto-discovery of chat, embedding, rerank, and vector_db plugins.
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Type

LLMPluginType = str  # "chat" | "embedding" | "rerank" | "vector_db"


class LLMRegistryError(RuntimeError):
    pass


LLM_REGISTRY: Dict[LLMPluginType, Dict[str, Type[Any]]] = {}
_DISCOVERED = False

_REQUIRED_METHOD: dict[str, str] = {
    "chat": "chat",
    "embedding": "embed",
    "rerank": "rerank",
    "vector_db": "search",
}

_SCAN_PACKAGES: tuple[str, ...] = (
    "fitz.llm.plugins",
    "fitz.llm.chat.plugins",
    "fitz.llm.embedding.plugins",
    "fitz.llm.rerank.plugins",
    "fitz.vector_db.plugins",
)


def get_llm_plugin(*, plugin_name: str, plugin_type: LLMPluginType) -> Type[Any]:
    """Get a plugin class by name and type."""
    _auto_discover()
    try:
        return LLM_REGISTRY[plugin_type][plugin_name]
    except KeyError as exc:
        available = list(LLM_REGISTRY.get(plugin_type, {}).keys())
        raise LLMRegistryError(
            f"Unknown {plugin_type} plugin: {plugin_name!r}. "
            f"Available: {available}"
        ) from exc


def available_llm_plugins(plugin_type: LLMPluginType) -> list[str]:
    """List available plugins for a given type."""
    _auto_discover()
    return sorted(LLM_REGISTRY.get(plugin_type, {}).keys())


def resolve_llm_plugin(
        *,
        plugin_type: LLMPluginType,
        requested_name: str,
) -> Type[Any]:
    """
    Resolve an LLM plugin by name.

    CHANGED: Now respects the requested plugin name instead of always
    falling back to local. This allows config to actually control which
    plugin is used.

    If the requested plugin is not available, falls back to local if available.

    Args:
        plugin_type: Type of plugin (chat, embedding, rerank, vector_db)
        requested_name: Name of the plugin to resolve

    Returns:
        Plugin class

    Raises:
        LLMRegistryError: If plugin not found and no fallback available
    """
    _auto_discover()

    bucket = LLM_REGISTRY.get(plugin_type, {})

    # 1. Try the explicitly requested plugin first
    if requested_name in bucket:
        return bucket[requested_name]

    # 2. If not found, try local fallback
    for cls in bucket.values():
        if getattr(cls, "availability", None) == "local":
            return cls

    # 3. No plugin found
    available = list(bucket.keys())
    raise LLMRegistryError(
        f"Unknown {plugin_type} plugin: {requested_name!r}. "
        f"Available: {available}"
    )


def _auto_discover() -> None:
    """Auto-discover plugins from scan packages."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    for pkg_name in _SCAN_PACKAGES:
        _scan_package_best_effort(pkg_name)

    _DISCOVERED = True


def _scan_package_best_effort(package_name: str) -> None:
    """Scan a package for plugins, ignoring errors."""
    try:
        pkg = importlib.import_module(package_name)
    except Exception:
        return

    pkg_path = getattr(pkg, "__path__", None)
    if pkg_path is None:
        return

    for module_info in pkgutil.iter_modules(pkg_path):
        module_name = f"{package_name}.{module_info.name}"
        try:
            mod = importlib.import_module(module_name)
        except Exception:
            continue

        # Look for plugin classes
        for attr_name in dir(mod):
            cls = getattr(mod, attr_name, None)
            if not isinstance(cls, type):
                continue

            p_name = getattr(cls, "plugin_name", None)
            p_type = getattr(cls, "plugin_type", None)

            if not p_name or not p_type:
                continue

            # Verify it has the required method
            required = _REQUIRED_METHOD.get(p_type)
            if required and not hasattr(cls, required):
                continue

            # Register it
            if p_type not in LLM_REGISTRY:
                LLM_REGISTRY[p_type] = {}
            LLM_REGISTRY[p_type][p_name] = cls
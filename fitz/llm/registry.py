# fitz/llm/registry.py
"""
Central LLM plugin registry.

Handles discovery and registration of:
- chat plugins
- embedding plugins
- rerank plugins

Note: vector_db plugins have their own separate registry at fitz.vector_db.registry

Design principle: NO SILENT FALLBACK
- If user configures "cohere", they get cohere or an error
- If user wants local, they explicitly configure "local"
- No magic substitution that could cause confusion
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Iterable, Type

from fitz.logging.logger import get_logger
from fitz.logging.tags import CHAT

logger = get_logger(__name__)

LLMPluginType = str  # "chat" | "embedding" | "rerank"


class LLMRegistryError(RuntimeError):
    pass


LLM_REGISTRY: Dict[LLMPluginType, Dict[str, Type[Any]]] = {}
_DISCOVERED = False

_REQUIRED_METHOD: dict[str, str] = {
    "chat": "chat",
    "embedding": "embed",
    "rerank": "rerank",
}

# Only LLM-related packages - vector_db has its own registry
_SCAN_PACKAGES: tuple[str, ...] = (
    "fitz.llm.chat.plugins",
    "fitz.llm.embedding.plugins",
    "fitz.llm.rerank.plugins",
)


def get_llm_plugin(*, plugin_name: str, plugin_type: LLMPluginType) -> Type[Any]:
    """
    Get an LLM plugin by exact name and type.

    No fallback, no magic - returns exactly what you ask for or raises an error.

    Args:
        plugin_name: Exact name of the plugin (e.g., "cohere", "openai", "local")
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        Plugin class

    Raises:
        LLMRegistryError: If plugin not found
    """
    _auto_discover()
    try:
        return LLM_REGISTRY[plugin_type][plugin_name]
    except KeyError as exc:
        available = sorted(LLM_REGISTRY.get(plugin_type, {}).keys())
        raise LLMRegistryError(
            f"Unknown {plugin_type} plugin: {plugin_name!r}. "
            f"Available: {available}"
        ) from exc


def available_llm_plugins(plugin_type: LLMPluginType) -> list[str]:
    """List available plugins for a given type."""
    _auto_discover()
    return sorted(LLM_REGISTRY.get(plugin_type, {}).keys())


# Alias for backwards compatibility - same as get_llm_plugin, no magic
def resolve_llm_plugin(
        *,
        plugin_type: LLMPluginType,
        requested_name: str,
) -> Type[Any]:
    """
    Resolve an LLM plugin by name.

    This is an alias for get_llm_plugin() for backwards compatibility.
    No fallback behavior - returns exactly what you request or raises an error.
    """
    return get_llm_plugin(plugin_name=requested_name, plugin_type=plugin_type)


def _auto_discover() -> None:
    """Discover all LLM plugins from scan packages."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    for pkg_name in _SCAN_PACKAGES:
        _scan_package_best_effort(pkg_name)

    logger.debug(f"{CHAT} Discovered LLM plugins: { {k: list(v.keys()) for k, v in LLM_REGISTRY.items()} }")
    _DISCOVERED = True


def _scan_package_best_effort(package_name: str) -> None:
    """Scan a package for plugin classes."""
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
            module = importlib.import_module(module_name)
        except Exception:
            continue

        for cls in _iter_plugin_classes(module, plugin_type=_infer_plugin_type(package_name)):
            _register(cls)


def _infer_plugin_type(package_name: str) -> str | None:
    """Infer plugin type from package name."""
    if "chat" in package_name:
        return "chat"
    if "embedding" in package_name:
        return "embedding"
    if "rerank" in package_name:
        return "rerank"
    return None


def _iter_plugin_classes(module: object, plugin_type: str | None) -> Iterable[type]:
    """Iterate over plugin classes in a module."""
    mod_name = getattr(module, "__name__", "")
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod_name:
            continue

        plugin_name = getattr(obj, "plugin_name", None)
        obj_plugin_type = getattr(obj, "plugin_type", None)

        if not isinstance(plugin_name, str) or not plugin_name:
            continue
        # Accept if plugin_type matches expected, or if not specified
        if plugin_type and obj_plugin_type != plugin_type:
            continue

        # Must have required method for this plugin type
        required = _REQUIRED_METHOD.get(obj_plugin_type or plugin_type or "")
        if required:
            fn = getattr(obj, required, None)
            if not callable(fn):
                continue

        yield obj


def _register(cls: Type[Any]) -> None:
    """Register a plugin class."""
    plugin_name = getattr(cls, "plugin_name")
    plugin_type = getattr(cls, "plugin_type")

    if plugin_type not in LLM_REGISTRY:
        LLM_REGISTRY[plugin_type] = {}

    existing = LLM_REGISTRY[plugin_type].get(plugin_name)
    if existing is not None and existing is not cls:
        raise LLMRegistryError(
            f"Duplicate {plugin_type} plugin_name={plugin_name!r}: "
            f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
        )

    LLM_REGISTRY[plugin_type][plugin_name] = cls
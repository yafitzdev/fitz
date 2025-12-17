# fitz/vector_db/registry.py
"""
Vector DB plugin registry.

Handles discovery and registration of vector database plugins.
Separate from LLM registry for cleaner architecture.
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Iterable, Type, TYPE_CHECKING

from fitz.logging.logger import get_logger
from fitz.logging.tags import VECTOR_DB

if TYPE_CHECKING:
    from fitz.vector_db.base import VectorDBPlugin

logger = get_logger(__name__)


class VectorDBRegistryError(RuntimeError):
    pass


VECTOR_DB_REGISTRY: Dict[str, Type[Any]] = {}
_DISCOVERED = False

_SCAN_PACKAGES: tuple[str, ...] = (
    "fitz.vector_db.plugins",
)


def get_vector_db_plugin(plugin_name: str) -> Type["VectorDBPlugin"]:
    """Get a vector DB plugin by name."""
    _auto_discover()
    try:
        return VECTOR_DB_REGISTRY[plugin_name]
    except KeyError as exc:
        available = sorted(VECTOR_DB_REGISTRY.keys())
        raise VectorDBRegistryError(
            f"Unknown vector_db plugin: {plugin_name!r}. "
            f"Available: {available}"
        ) from exc


def available_vector_db_plugins() -> list[str]:
    """List available vector DB plugins."""
    _auto_discover()
    return sorted(VECTOR_DB_REGISTRY.keys())


def resolve_vector_db_plugin(requested_name: str) -> Type["VectorDBPlugin"]:
    """
    Resolve a vector DB plugin with local-first fallback.

    Resolution order:
    1. Any plugin with availability="local"
    2. The explicitly requested plugin_name
    """
    _auto_discover()

    # 1. local-first
    for cls in VECTOR_DB_REGISTRY.values():
        if getattr(cls, "availability", None) == "local":
            return cls

    # 2. explicit request
    try:
        return VECTOR_DB_REGISTRY[requested_name]
    except KeyError as exc:
        available = sorted(VECTOR_DB_REGISTRY.keys())
        raise VectorDBRegistryError(
            f"Unknown vector_db plugin: {requested_name!r}. "
            f"Available: {available}"
        ) from exc


def _auto_discover() -> None:
    """Discover all vector DB plugins from scan packages."""
    global _DISCOVERED
    if _DISCOVERED:
        return

    for pkg_name in _SCAN_PACKAGES:
        _scan_package_best_effort(pkg_name)

    logger.debug(f"{VECTOR_DB} Discovered vector_db plugins: {list(VECTOR_DB_REGISTRY.keys())}")
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

        for cls in _iter_plugin_classes(module):
            _register(cls)


def _iter_plugin_classes(module: object) -> Iterable[type]:
    """Iterate over plugin classes in a module."""
    mod_name = getattr(module, "__name__", "")
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod_name:
            continue

        plugin_name = getattr(obj, "plugin_name", None)
        plugin_type = getattr(obj, "plugin_type", None)

        if not isinstance(plugin_name, str) or not plugin_name:
            continue
        # Only accept vector_db plugins
        if plugin_type != "vector_db":
            continue

        # Must have search method
        fn = getattr(obj, "search", None)
        if not callable(fn):
            continue

        yield obj


def _register(cls: Type[Any]) -> None:
    """Register a plugin class."""
    plugin_name = getattr(cls, "plugin_name")

    existing = VECTOR_DB_REGISTRY.get(plugin_name)
    if existing is not None and existing is not cls:
        raise VectorDBRegistryError(
            f"Duplicate vector_db plugin_name={plugin_name!r}: "
            f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
        )

    VECTOR_DB_REGISTRY[plugin_name] = cls
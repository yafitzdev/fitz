from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, Type

from core.logging.logger import get_logger
from core.logging.tags import VECTOR_DB

logger = get_logger(__name__)


@dataclass(frozen=True)
class VectorDBPluginKey:
    name: str


# plugin_name → plugin class
_VECTOR_DB_PLUGINS: Dict[VectorDBPluginKey, Type[Any]] = {}
_DISCOVERY_DONE: bool = False


class VectorDBRegistryError(RuntimeError):
    """Base error for vector DB plugin registry."""


def register_vector_db_plugin(plugin_cls: Type[Any], *, plugin_name: str) -> None:
    """
    Register a single vector DB plugin class.

    This is intended to be called from plugin modules, e.g.:

        register_vector_db_plugin(QdrantVectorDB, plugin_name="qdrant")
    """
    key = VectorDBPluginKey(plugin_name)

    if key in _VECTOR_DB_PLUGINS:
        logger.info(
            f"{VECTOR_DB} Overwriting vector DB plugin registration "
            f"for name='{plugin_name}'"
        )

    _VECTOR_DB_PLUGINS[key] = plugin_cls
    logger.info(
        f"{VECTOR_DB} Registered vector DB plugin "
        f"name='{plugin_name}', cls='{plugin_cls.__name__}'"
    )


def get_vector_db_plugin(plugin_name: str) -> Type[Any]:
    """
    Return the plugin class for `plugin_name`.

    Example:
        QdrantCls = get_vector_db_plugin("qdrant")
        client = QdrantCls(host="localhost", port=6333)
    """
    if not _DISCOVERY_DONE:
        auto_discover_vector_db_plugins()

    key = VectorDBPluginKey(plugin_name)
    try:
        return _VECTOR_DB_PLUGINS[key]
    except KeyError as exc:
        available = ", ".join(k.name for k in _VECTOR_DB_PLUGINS.keys()) or "<none>"
        raise VectorDBRegistryError(
            f"No vector DB plugin registered for name='{plugin_name}'. "
            f"Available: {available}"
        ) from exc


def auto_discover_vector_db_plugins() -> None:
    """
    Import all modules under known vector DB plugin namespaces so that
    plugins can self-register via `register_vector_db_plugin()`.
    """
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return

    namespaces = [
        "fitz_stack.vector_db.plugins",
    ]

    for ns in namespaces:
        try:
            pkg = importlib.import_module(ns)
        except ModuleNotFoundError:
            # Namespace might legitimately not exist yet.
            continue

        if not hasattr(pkg, "__path__"):
            continue

        for module_info in pkgutil.iter_modules(pkg.__path__, prefix=f"{ns}."):
            try:
                importlib.import_module(module_info.name)
                logger.info(
                    f"{VECTOR_DB} Auto-discovered vector DB plugin "
                    f"module '{module_info.name}'"
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.info(
                    f"{VECTOR_DB} Failed to import vector DB plugin module "
                    f"'{module_info.name}': {exc}"
                )

    _DISCOVERY_DONE = True

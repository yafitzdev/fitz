# ============================
# File: core/llm/registry.py
# ============================
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass
from typing import Any, Dict, Literal, Type

from core.logging import get_logger
from core.logging_tags import PIPELINE

logger = get_logger(__name__)

PluginType = Literal["embedding", "rerank", "chat"]


@dataclass(frozen=True)
class LLMPluginKey:
    name: str
    type: PluginType


_LLM_PLUGINS: Dict[LLMPluginKey, Type[Any]] = {}
_DISCOVERY_DONE: bool = False


class LLMRegistryError(RuntimeError):
    pass


def register_llm_plugin(
    plugin_cls: Type[Any],
    *,
    plugin_name: str,
    plugin_type: PluginType,
) -> None:
    key = LLMPluginKey(plugin_name, plugin_type)

    if key in _LLM_PLUGINS:
        logger.info(
            f"{PIPELINE} Overwriting LLM plugin registration for "
            f"name='{plugin_name}', type='{plugin_type}'"
        )

    _LLM_PLUGINS[key] = plugin_cls
    logger.info(
        f"{PIPELINE} Registered LLM plugin name='{plugin_name}', "
        f"type='{plugin_type}', cls='{plugin_cls.__name__}'"
    )


def get_llm_plugin(plugin_name: str, plugin_type: PluginType) -> Type[Any]:
    if not _DISCOVERY_DONE:
        auto_discover_llm_plugins()

    key = LLMPluginKey(plugin_name, plugin_type)
    try:
        return _LLM_PLUGINS[key]
    except KeyError as exc:
        available = ", ".join(f"{k.name}:{k.type}" for k in _LLM_PLUGINS.keys()) or "<none>"
        raise LLMRegistryError(
            f"No LLM plugin registered for name='{plugin_name}', type='{plugin_type}'. "
            f"Available: {available}"
        ) from exc


def auto_discover_llm_plugins() -> None:
    global _DISCOVERY_DONE
    if _DISCOVERY_DONE:
        return

    namespaces = [
        "core.llm.embedding.plugins",
        "core.llm.rerank.plugins",
        "core.llm.chat.plugins",
    ]

    for ns in namespaces:
        try:
            pkg = importlib.import_module(ns)
        except ModuleNotFoundError:
            continue

        if not hasattr(pkg, "__path__"):
            continue

        for module_info in pkgutil.iter_modules(pkg.__path__, prefix=f"{ns}."):
            try:
                importlib.import_module(module_info.name)
                logger.info(
                    f"{PIPELINE} Auto-discovered LLM plugin module '{module_info.name}'"
                )
            except Exception as exc:
                logger.info(
                    f"{PIPELINE} Failed to import LLM plugin module '{module_info.name}': {exc}"
                )

    _DISCOVERY_DONE = True

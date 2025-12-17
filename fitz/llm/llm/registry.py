# core/llm/registry.py
from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, Iterable, Type

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
    "fitz.core.llm.plugins",
    "fitz.core.llm.chat.plugins",
    "fitz.core.llm.embedding.plugins",
    "fitz.core.llm.rerank.plugins",
    "fitz.core.vector_db.plugins",
)


def get_llm_plugin(*, plugin_name: str, plugin_type: LLMPluginType) -> Type[Any]:
    _auto_discover()
    try:
        return LLM_REGISTRY[plugin_type][plugin_name]
    except KeyError as exc:
        raise LLMRegistryError(f"Unknown {plugin_type} plugin: {plugin_name!r}") from exc


def available_llm_plugins(plugin_type: LLMPluginType) -> list[str]:
    _auto_discover()
    return sorted(LLM_REGISTRY.get(plugin_type, {}).keys())


def resolve_llm_plugin(
    *,
    plugin_type: LLMPluginType,
    requested_name: str,
) -> Type[Any]:
    """
    Resolve an LLM plugin with local-first fallback.

    Resolution order:
    1. Any plugin with availability="local" for this plugin_type
    2. The explicitly requested plugin_name
    """
    _auto_discover()

    bucket = LLM_REGISTRY.get(plugin_type, {})

    # 1. local-first
    for cls in bucket.values():
        if getattr(cls, "availability", None) == "local":
            return cls

    # 2. explicit request
    try:
        return bucket[requested_name]
    except KeyError as exc:
        raise LLMRegistryError(f"Unknown {plugin_type} plugin: {requested_name!r}") from exc


def _auto_discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    for pkg_name in _SCAN_PACKAGES:
        _scan_package_best_effort(pkg_name)

    _DISCOVERED = True


def _scan_package_best_effort(package_name: str) -> None:
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
        if not isinstance(plugin_type, str) or not plugin_type:
            continue

        required = _REQUIRED_METHOD.get(plugin_type)
        if required is None:
            continue

        fn = getattr(obj, required, None)
        if not callable(fn):
            continue

        yield obj


def _register(cls: Type[Any]) -> None:
    plugin_name = getattr(cls, "plugin_name")
    plugin_type = getattr(cls, "plugin_type")

    bucket = LLM_REGISTRY.setdefault(plugin_type, {})
    existing = bucket.get(plugin_name)

    if existing is not None and existing is not cls:
        raise LLMRegistryError(
            f"Duplicate {plugin_type} plugin_name={plugin_name!r}: "
            f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
        )

    bucket[plugin_name] = cls

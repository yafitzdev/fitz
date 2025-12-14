# pipeline/pipeline/registry.py
from __future__ import annotations

import importlib
import pkgutil
from typing import Dict, Iterable, Type

from fitz.pipeline.pipeline.base import PipelinePlugin

PIPELINE_REGISTRY: Dict[str, Type[PipelinePlugin]] = {}
_DISCOVERED = False


def _iter_plugin_classes(module: object) -> Iterable[type]:
    mod_name = getattr(module, "__name__", "")
    for obj in vars(module).values():
        if not isinstance(obj, type):
            continue
        if getattr(obj, "__module__", None) != mod_name:
            continue

        plugin_name = getattr(obj, "plugin_name", None)
        if not isinstance(plugin_name, str) or not plugin_name:
            continue

        build_fn = getattr(obj, "build", None)
        if not callable(build_fn):
            continue

        yield obj


def _auto_discover() -> None:
    global _DISCOVERED
    if _DISCOVERED:
        return

    plugins_pkg = importlib.import_module("fitz.pipeline.pipeline.plugins")

    for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
        module = importlib.import_module(f"{plugins_pkg.__name__}.{module_info.name}")

        for cls in _iter_plugin_classes(module):
            name = getattr(cls, "plugin_name")
            existing = PIPELINE_REGISTRY.get(name)
            if existing is not None and existing is not cls:
                raise ValueError(
                    f"Duplicate pipeline plugin_name={name!r}: "
                    f"{existing.__module__}.{existing.__name__} vs {cls.__module__}.{cls.__name__}"
                )
            PIPELINE_REGISTRY[name] = cls  # type: ignore[assignment]

    _DISCOVERED = True


def get_pipeline_plugin(name: str) -> Type[PipelinePlugin]:
    _auto_discover()
    try:
        return PIPELINE_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown pipeline plugin: {name!r}") from exc


def available_pipeline_plugins() -> list[str]:
    _auto_discover()
    return sorted(PIPELINE_REGISTRY.keys())

# fitz/llm/loader.py
"""
YAML plugin loader with schema validation.

Loads YAML plugin definitions and validates them against Pydantic schemas.
Invalid plugins fail fast with clear error messages.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, overload

import yaml
from pydantic import ValidationError

from fitz_ai.llm.schema import (
    ChatPluginSpec,
    EmbeddingPluginSpec,
    PluginSpec,
    RerankPluginSpec,
)

logger = logging.getLogger(__name__)

YAML_PLUGINS_DIR = Path(__file__).parent

_SPEC_CLASSES: dict[str, type[PluginSpec]] = {
    "chat": ChatPluginSpec,
    "embedding": EmbeddingPluginSpec,
    "rerank": RerankPluginSpec,
}


class YAMLPluginError(Exception):
    """Base error for YAML plugin operations."""

    pass


class YAMLPluginNotFoundError(YAMLPluginError, FileNotFoundError):
    """YAML plugin file not found."""

    pass


class YAMLPluginValidationError(YAMLPluginError, ValueError):
    """YAML plugin failed schema validation."""

    def __init__(self, plugin_path: Path, errors: list[dict]):
        self.plugin_path = plugin_path
        self.errors = errors

        error_lines = [f"Invalid YAML plugin: {plugin_path}"]
        for err in errors:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            error_lines.append(f"  - {loc}: {msg}")

        super().__init__("\n".join(error_lines))


def _get_yaml_path(plugin_type: str, plugin_name: str) -> Path:
    """Get the path to a YAML plugin file."""
    return YAML_PLUGINS_DIR / plugin_type / f"{plugin_name}.yaml"


def _load_yaml_file(path: Path) -> dict:
    """Load and parse a YAML file."""
    if not path.exists():
        raise YAMLPluginNotFoundError(f"YAML plugin not found: {path}")

    with open(path) as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise YAMLPluginError(f"Invalid YAML syntax in {path}: {e}") from e

    if not isinstance(data, dict):
        raise YAMLPluginError(f"YAML plugin must be a mapping, got {type(data).__name__}")

    return data


@lru_cache(maxsize=256)
def _load_plugin_cached(plugin_type: str, plugin_name: str) -> PluginSpec:
    """Load and validate a plugin specification (cached)."""
    if plugin_type not in _SPEC_CLASSES:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(_SPEC_CLASSES.keys())}"
        )

    spec_class = _SPEC_CLASSES[plugin_type]
    path = _get_yaml_path(plugin_type, plugin_name)
    data = _load_yaml_file(path)

    try:
        return spec_class.model_validate(data)
    except ValidationError as e:
        raise YAMLPluginValidationError(path, e.errors()) from e


@overload
def load_plugin(plugin_type: Literal["chat"], plugin_name: str) -> ChatPluginSpec: ...


@overload
def load_plugin(plugin_type: Literal["embedding"], plugin_name: str) -> EmbeddingPluginSpec: ...


@overload
def load_plugin(plugin_type: Literal["rerank"], plugin_name: str) -> RerankPluginSpec: ...


@overload
def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec: ...


def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec:
    """
    Load and validate a plugin specification.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated PluginSpec
    """
    return _load_plugin_cached(plugin_type, plugin_name)


def list_yaml_plugins(plugin_type: str) -> list[str]:
    """
    List available YAML plugins for a type.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        Sorted list of plugin names
    """
    if plugin_type not in _SPEC_CLASSES:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(_SPEC_CLASSES.keys())}"
        )

    plugin_dir = YAML_PLUGINS_DIR / plugin_type
    if not plugin_dir.exists():
        return []

    return sorted(p.stem for p in plugin_dir.glob("*.yaml") if not p.stem.startswith("_"))


def clear_cache() -> None:
    """Clear the plugin loading cache."""
    _load_plugin_cached.cache_clear()


__all__ = [
    "load_plugin",
    "list_yaml_plugins",
    "clear_cache",
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
]

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

from fitz.llm.schema import (
    ChatPluginSpec,
    EmbeddingPluginSpec,
    PluginSpec,
    RerankPluginSpec,
)

logger = logging.getLogger(__name__)

# Directory containing YAML plugin definitions (fitz/llm/)
YAML_PLUGINS_DIR = Path(__file__).parent

# Mapping of plugin types to their spec classes
_SPEC_CLASSES: dict[str, type[PluginSpec]] = {
    "chat": ChatPluginSpec,
    "embedding": EmbeddingPluginSpec,
    "rerank": RerankPluginSpec,
}


# =============================================================================
# Exceptions
# =============================================================================


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

        # Format errors nicely
        error_lines = [f"Invalid YAML plugin: {plugin_path}"]
        for err in errors:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            error_lines.append(f"  - {loc}: {msg}")

        super().__init__("\n".join(error_lines))


# =============================================================================
# Internal Helpers
# =============================================================================


def _get_yaml_path(plugin_type: str, plugin_name: str) -> Path:
    """Get the path to a YAML plugin file.

    Structure: fitz/llm/{plugin_type}/{plugin_name}.yaml
    Example: fitz/llm/chat/openai.yaml
    """
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


# =============================================================================
# Core Generic Loader (Single Implementation)
# =============================================================================


@lru_cache(maxsize=256)
def _load_plugin_cached(plugin_type: str, plugin_name: str) -> PluginSpec:
    """
    Load and validate a plugin specification (cached).

    This is the single implementation for all plugin types.
    The cache key is (plugin_type, plugin_name).

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated PluginSpec

    Raises:
        ValueError: If plugin_type is invalid
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation
    """
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


# =============================================================================
# Type-Safe Public API
# =============================================================================


@overload
def load_plugin(plugin_type: Literal["chat"], plugin_name: str) -> ChatPluginSpec: ...

@overload
def load_plugin(plugin_type: Literal["embedding"], plugin_name: str) -> EmbeddingPluginSpec: ...

@overload
def load_plugin(plugin_type: Literal["rerank"], plugin_name: str) -> RerankPluginSpec: ...

@overload
def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec: ...


def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec:
    """Load and validate any plugin specification.

    This is the primary entry point for loading plugins. It returns
    the appropriate spec type based on the plugin_type.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated PluginSpec (ChatPluginSpec, EmbeddingPluginSpec, or RerankPluginSpec)

    Raises:
        ValueError: If plugin_type is invalid
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation

    Examples:
        >>> spec = load_plugin("chat", "openai")
        >>> spec.plugin_name
        'openai'
    """
    return _load_plugin_cached(plugin_type, plugin_name)


def load_chat_plugin(plugin_name: str) -> ChatPluginSpec:
    """Load and validate a chat plugin specification.

    Type-safe wrapper around load_plugin() for chat plugins.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated ChatPluginSpec
    """
    return _load_plugin_cached("chat", plugin_name)  # type: ignore[return-value]


def load_embedding_plugin(plugin_name: str) -> EmbeddingPluginSpec:
    """Load and validate an embedding plugin specification.

    Type-safe wrapper around load_plugin() for embedding plugins.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated EmbeddingPluginSpec
    """
    return _load_plugin_cached("embedding", plugin_name)  # type: ignore[return-value]


def load_rerank_plugin(plugin_name: str) -> RerankPluginSpec:
    """Load and validate a rerank plugin specification.

    Type-safe wrapper around load_plugin() for rerank plugins.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere")

    Returns:
        Validated RerankPluginSpec
    """
    return _load_plugin_cached("rerank", plugin_name)  # type: ignore[return-value]


# =============================================================================
# Utility Functions
# =============================================================================


def list_yaml_plugins(plugin_type: str) -> list[str]:
    """List available YAML plugins for a type.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        Sorted list of plugin names

    Examples:
        >>> list_yaml_plugins("chat")
        ['anthropic', 'cohere', 'openai', ...]
    """
    if plugin_type not in _SPEC_CLASSES:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(_SPEC_CLASSES.keys())}"
        )

    plugin_dir = YAML_PLUGINS_DIR / plugin_type

    if not plugin_dir.exists():
        return []

    return sorted(
        p.stem for p in plugin_dir.glob("*.yaml")
        if not p.stem.startswith("_")  # Ignore files like _template.yaml
    )


def clear_cache() -> None:
    """Clear the plugin loading cache.

    Useful for testing or when YAML files are modified.
    """
    _load_plugin_cached.cache_clear()


__all__ = [
    # Primary API
    "load_plugin",
    # Type-safe wrappers
    "load_chat_plugin",
    "load_embedding_plugin",
    "load_rerank_plugin",
    # Utilities
    "list_yaml_plugins",
    "clear_cache",
    # Exceptions
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
]
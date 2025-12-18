# fitz/llm/yaml_plugins/loader.py
"""
YAML plugin loader with schema validation.

Loads YAML plugin definitions and validates them against Pydantic schemas.
Invalid plugins fail fast with clear error messages.
"""
from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import TypeVar

import yaml
from pydantic import ValidationError

from fitz.llm.yaml_plugins.schema import (
    ChatPluginSpec,
    EmbeddingPluginSpec,
    PluginSpec,
    RerankPluginSpec,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=PluginSpec)

# Directory containing YAML plugin definitions
YAML_PLUGINS_DIR = Path(__file__).parent


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
# Plugin Loading
# =============================================================================


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


def _validate_spec(data: dict, spec_class: type[T], path: Path) -> T:
    """Validate YAML data against a Pydantic schema."""
    try:
        return spec_class.model_validate(data)
    except ValidationError as e:
        raise YAMLPluginValidationError(path, e.errors()) from e


# =============================================================================
# Public API
# =============================================================================


@lru_cache(maxsize=128)
def load_chat_plugin(plugin_name: str) -> ChatPluginSpec:
    """Load and validate a chat plugin specification.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated ChatPluginSpec

    Raises:
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation
    """
    path = _get_yaml_path("chat", plugin_name)
    data = _load_yaml_file(path)
    return _validate_spec(data, ChatPluginSpec, path)


@lru_cache(maxsize=128)
def load_embedding_plugin(plugin_name: str) -> EmbeddingPluginSpec:
    """Load and validate an embedding plugin specification.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere", "openai")

    Returns:
        Validated EmbeddingPluginSpec

    Raises:
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation
    """
    path = _get_yaml_path("embedding", plugin_name)
    data = _load_yaml_file(path)
    return _validate_spec(data, EmbeddingPluginSpec, path)


@lru_cache(maxsize=128)
def load_rerank_plugin(plugin_name: str) -> RerankPluginSpec:
    """Load and validate a rerank plugin specification.

    Args:
        plugin_name: Name of the plugin (e.g., "cohere")

    Returns:
        Validated RerankPluginSpec

    Raises:
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation
    """
    path = _get_yaml_path("rerank", plugin_name)
    data = _load_yaml_file(path)
    return _validate_spec(data, RerankPluginSpec, path)


def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec:
    """Load and validate any plugin specification.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")
        plugin_name: Name of the plugin

    Returns:
        Validated PluginSpec (ChatPluginSpec, EmbeddingPluginSpec, or RerankPluginSpec)

    Raises:
        ValueError: If plugin_type is invalid
        YAMLPluginNotFoundError: If plugin file doesn't exist
        YAMLPluginValidationError: If plugin fails validation
    """
    loaders = {
        "chat": load_chat_plugin,
        "embedding": load_embedding_plugin,
        "rerank": load_rerank_plugin,
    }

    if plugin_type not in loaders:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(loaders.keys())}"
        )

    return loaders[plugin_type](plugin_name)


def list_yaml_plugins(plugin_type: str) -> list[str]:
    """List available YAML plugins for a type.

    Args:
        plugin_type: Type of plugin ("chat", "embedding", "rerank")

    Returns:
        List of plugin names
    """
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
    load_chat_plugin.cache_clear()
    load_embedding_plugin.cache_clear()
    load_rerank_plugin.cache_clear()
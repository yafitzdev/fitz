# fitz_ai/llm/loader.py
"""
YAML plugin loader with schema validation.

Loads YAML plugin definitions and validates them against:
1. Pydantic schemas (type checking)
2. Master schema files (field completeness)

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

        error_lines = [f"Invalid YAML plugin: {plugin_path}"]
        for err in errors:
            loc = " -> ".join(str(x) for x in err.get("loc", []))
            msg = err.get("msg", "Unknown error")
            error_lines.append(f"  - {loc}: {msg}")

        super().__init__("\n".join(error_lines))


# =============================================================================
# Path Resolution
# =============================================================================


def _get_yaml_path(plugin_type: str, plugin_name: str) -> Path:
    """Get the path to a YAML plugin file."""
    return YAML_PLUGINS_DIR / plugin_type / f"{plugin_name}.yaml"


# =============================================================================
# YAML Loading
# =============================================================================


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
        raise YAMLPluginError(
            f"YAML plugin must be a mapping, got {type(data).__name__}"
        )

    return data


# =============================================================================
# Schema Defaults Application
# =============================================================================


def _apply_defaults(data: dict, plugin_type: str) -> dict:
    """
    Apply defaults from master schema to plugin data.

    Missing optional fields get their default values from the schema YAML.
    """
    try:
        from fitz_ai.llm.schema_defaults import get_nested_defaults

        defaults = get_nested_defaults(plugin_type)
    except (ImportError, FileNotFoundError):
        # Schema files not available, skip defaults
        return data

    return _deep_merge(defaults, data)


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """
    Deep merge two dicts. Values in overrides take precedence.
    """
    result = dict(defaults)

    for key, value in overrides.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


# =============================================================================
# Schema Validation
# =============================================================================


def _validate_against_master_schema(
    data: dict, plugin_type: str, path: Path
) -> list[str]:
    """
    Validate plugin data against master schema.

    Returns list of warning messages (non-fatal).
    """
    warnings = []

    try:
        from fitz_ai.llm.schema_defaults import validate_plugin_fields

        errors = validate_plugin_fields(plugin_type, data, strict=False)

        for error in errors:
            if "Missing required field" in error:
                # Required field missing is a hard error, will be caught by Pydantic
                pass
            else:
                warnings.append(error)

    except (ImportError, FileNotFoundError):
        # Schema files not available, skip validation
        pass

    return warnings


# =============================================================================
# Plugin Loading
# =============================================================================


@lru_cache(maxsize=256)
def _load_plugin_cached(plugin_type: str, plugin_name: str) -> PluginSpec:
    """Load and validate a plugin specification (cached)."""
    if plugin_type not in _SPEC_CLASSES:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(_SPEC_CLASSES.keys())}"
        )

    yaml_path = _get_yaml_path(plugin_type, plugin_name)
    raw_data = _load_yaml_file(yaml_path)

    # Apply defaults from master schema
    data = _apply_defaults(raw_data, plugin_type)

    # Validate against master schema (warnings only)
    warnings = _validate_against_master_schema(data, plugin_type, yaml_path)
    for warning in warnings:
        logger.warning(f"Plugin {plugin_name}: {warning}")

    # Validate with Pydantic
    spec_class = _SPEC_CLASSES[plugin_type]

    try:
        spec = spec_class.model_validate(data)
    except ValidationError as e:
        raise YAMLPluginValidationError(yaml_path, e.errors()) from e

    return spec


# =============================================================================
# Public API
# =============================================================================


@overload
def load_plugin(plugin_type: Literal["chat"], plugin_name: str) -> ChatPluginSpec: ...


@overload
def load_plugin(
    plugin_type: Literal["embedding"], plugin_name: str
) -> EmbeddingPluginSpec: ...


@overload
def load_plugin(
    plugin_type: Literal["rerank"], plugin_name: str
) -> RerankPluginSpec: ...


@overload
def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec: ...


def load_plugin(plugin_type: str, plugin_name: str) -> PluginSpec:
    """
    Load a plugin specification from YAML.

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        plugin_name: Plugin name (e.g., "openai", "cohere")

    Returns:
        Validated PluginSpec instance

    Raises:
        YAMLPluginNotFoundError: If YAML file doesn't exist
        YAMLPluginValidationError: If validation fails

    Example:
        >>> spec = load_plugin("chat", "openai")
        >>> spec.provider.base_url
        'https://api.openai.com/v1'
    """
    return _load_plugin_cached(plugin_type, plugin_name)


def list_plugins(plugin_type: str) -> list[str]:
    """
    List available plugins of a given type.

    Args:
        plugin_type: "chat", "embedding", or "rerank"

    Returns:
        List of plugin names (without .yaml extension)
    """
    plugins_dir = YAML_PLUGINS_DIR / plugin_type

    if not plugins_dir.exists():
        return []

    return sorted(f.stem for f in plugins_dir.glob("*.yaml"))


def clear_cache() -> None:
    """Clear the plugin loading cache."""
    _load_plugin_cached.cache_clear()


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "YAMLPluginError",
    "YAMLPluginNotFoundError",
    "YAMLPluginValidationError",
    # Functions
    "load_plugin",
    "list_plugins",
    "clear_cache",
]

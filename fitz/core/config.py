# fitz/core/config.py
"""
Centralized configuration loading for Fitz.

This module provides a SINGLE way to load configuration files across all
components. Individual config schemas remain in their respective packages,
but all loading logic is consolidated here.

Usage:
    from fitz.core.config import load_yaml, load_config, ConfigError

    # Load raw YAML
    data = load_yaml("config.yaml")

    # Load and validate with a schema
    from fitz.engines.classic_rag.config.schema import ClassicRagConfig
    config = load_config("config.yaml", ClassicRagConfig)

    # Auto-detect config type
    config = load_config("config.yaml")  # Returns appropriate type

Design principles:
    - Single source of truth for config loading
    - Schema validation is separate from loading
    - Clear error messages with file paths
    - Support for default configs
    - FitzPaths integration for standard locations
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml

from fitz.core.paths import FitzPaths

logger = logging.getLogger(__name__)

# Type variable for config classes
T = TypeVar("T")


# =============================================================================
# Errors
# =============================================================================


class ConfigError(Exception):
    """Base error for configuration issues."""

    def __init__(self, message: str, path: Optional[Path] = None):
        self.path = path
        if path:
            message = f"{message} (file: {path})"
        super().__init__(message)


class ConfigNotFoundError(ConfigError):
    """Raised when a config file doesn't exist."""

    pass


class ConfigParseError(ConfigError):
    """Raised when YAML parsing fails."""

    pass


class ConfigValidationError(ConfigError):
    """Raised when config doesn't match schema."""

    pass


# =============================================================================
# Core Loading Functions
# =============================================================================


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load a YAML file and return as dictionary.

    This is the lowest-level loading function. For most use cases,
    prefer load_config() which also validates.

    Args:
        path: Path to YAML file

    Returns:
        Dictionary of config data

    Raises:
        ConfigNotFoundError: If file doesn't exist
        ConfigParseError: If YAML is invalid
    """
    p = Path(path)

    if not p.exists():
        raise ConfigNotFoundError("Config file not found", path=p)

    if p.is_dir():
        raise ConfigError("Config path is a directory, not a file", path=p)

    try:
        with p.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ConfigParseError(f"Invalid YAML syntax: {e}", path=p) from e
    except Exception as e:
        raise ConfigParseError(f"Failed to read config: {e}", path=p) from e

    if not isinstance(data, dict):
        raise ConfigParseError("Config root must be a mapping (dict)", path=p)

    logger.debug(f"Loaded config from {p}")
    return data


def load_config(
    path: Optional[Union[str, Path]] = None,
    schema: Optional[Type[T]] = None,
    config_type: Optional[str] = None,
) -> T:
    """
    Load and validate a configuration file.

    Args:
        path: Path to config file. If None, uses default based on config_type.
        schema: Pydantic model or dataclass to validate against.
               If None, auto-detects based on config_type or file content.
        config_type: Type of config ("rag", "ingest", "clara").
                    Used to find defaults and auto-detect schema.

    Returns:
        Validated config object

    Raises:
        ConfigNotFoundError: If file doesn't exist
        ConfigParseError: If YAML is invalid
        ConfigValidationError: If config doesn't match schema

    Examples:
        # Load RAG config with explicit schema
        from fitz.engines.classic_rag.config.schema import ClassicRagConfig
        config = load_config("config.yaml", ClassicRagConfig)

        # Load default RAG config
        config = load_config(config_type="rag")

        # Load ingest config
        config = load_config("ingest.yaml", config_type="ingest")
    """
    # Resolve path
    resolved_path = _resolve_config_path(path, config_type)

    # Load raw YAML
    data = load_yaml(resolved_path)

    # Determine schema
    if schema is None:
        schema = _detect_schema(data, config_type)

    if schema is None:
        # No schema - return raw dict
        return data  # type: ignore

    # Validate with schema
    return _validate_config(data, schema, resolved_path)


def load_config_dict(path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    """
    Load configuration as raw dictionary (no validation).

    Useful for:
    - Inspecting config before deciding which schema to use
    - Advanced use cases where raw access is needed
    - CLI commands that just display config

    Args:
        path: Path to config file

    Returns:
        Raw config dictionary
    """
    if path is None:
        path = FitzPaths.config()
    return load_yaml(path)


# =============================================================================
# Default Config Paths
# =============================================================================


def get_default_config_path(config_type: str) -> Path:
    """
    Get the default config path for a given config type.

    Args:
        config_type: One of "rag", "ingest", "clara"

    Returns:
        Path to the default config file

    Raises:
        ValueError: If config_type is unknown
    """
    if config_type == "rag" or config_type == "classic_rag":
        # Classic RAG default is bundled with the engine
        from fitz.engines.classic_rag.config.loader import DEFAULT_CONFIG_PATH

        return DEFAULT_CONFIG_PATH

    elif config_type == "ingest":
        # Ingest doesn't have a bundled default - use workspace
        return FitzPaths.config_dir() / "ingest.yaml"

    elif config_type == "clara":
        # CLaRa default
        return FitzPaths.config_dir() / "clara.yaml"

    else:
        raise ValueError(f"Unknown config type: {config_type!r}")


def _resolve_config_path(
    path: Optional[Union[str, Path]],
    config_type: Optional[str],
) -> Path:
    """Resolve the actual config path to load."""
    if path is not None:
        return Path(path)

    if config_type is not None:
        return get_default_config_path(config_type)

    # Fall back to workspace config
    workspace_config = FitzPaths.config()
    if workspace_config.exists():
        return workspace_config

    # Try the bundled RAG default
    try:
        from fitz.engines.classic_rag.config.loader import DEFAULT_CONFIG_PATH

        if DEFAULT_CONFIG_PATH.exists():
            return DEFAULT_CONFIG_PATH
    except ImportError:
        pass

    raise ConfigNotFoundError(
        "No config file specified and no default found. " "Run 'fitz init' to create one."
    )


# =============================================================================
# Schema Detection and Validation
# =============================================================================


def _detect_schema(data: Dict[str, Any], config_type: Optional[str]) -> Optional[Type]:
    """
    Auto-detect the appropriate schema for config data.

    Uses config_type hint if provided, otherwise inspects data keys.

    Note: All imports are lazy to avoid circular dependencies and
    architecture violations (core should not import from engines/ingest at module level).
    """
    # Use explicit type hint
    if config_type == "rag" or config_type == "classic_rag":
        from fitz.engines.classic_rag.config.schema import ClassicRagConfig

        return ClassicRagConfig

    if config_type == "ingest":
        from fitz.ingest.config.schema import IngestConfig

        return IngestConfig

    if config_type == "clara":
        from fitz.engines.clara.config.schema import ClaraConfig

        return ClaraConfig

    # Auto-detect from content (lazy imports to avoid architecture violations)
    if "ingester" in data and "chunker" in data:
        from fitz.ingest.config.schema import IngestConfig

        return IngestConfig

    if "chat" in data or "retriever" in data or "vector_db" in data:
        from fitz.engines.classic_rag.config.schema import ClassicRagConfig

        return ClassicRagConfig

    if "model" in data and "compression" in data:
        from fitz.engines.clara.config.schema import ClaraConfig

        return ClaraConfig

    # Unknown - return None to get raw dict
    return None


def _validate_config(
    data: Dict[str, Any],
    schema: Type[T],
    path: Path,
) -> T:
    """
    Validate config data against a schema.

    Supports both Pydantic models and dataclasses with from_dict().
    """
    try:
        # Try Pydantic model_validate first
        if hasattr(schema, "model_validate"):
            return schema.model_validate(data)

        # Try from_dict (for dataclasses)
        if hasattr(schema, "from_dict"):
            return schema.from_dict(data)

        # Try direct instantiation
        return schema(**data)

    except Exception as e:
        raise ConfigValidationError(
            f"Config validation failed: {e}",
            path=path,
        ) from e


# =============================================================================
# Convenience Functions
# =============================================================================


def load_rag_config(path: Optional[Union[str, Path]] = None):
    """
    Load Classic RAG configuration.

    Convenience wrapper for load_config(path, config_type="rag").
    """
    # Lazy import to avoid architecture violation
    from fitz.engines.classic_rag.config.schema import ClassicRagConfig

    return load_config(path, schema=ClassicRagConfig, config_type="rag")


def load_ingest_config(path: Union[str, Path]):
    """
    Load ingestion configuration.

    Convenience wrapper for load_config(path, config_type="ingest").
    """
    # Lazy import to avoid architecture violation
    from fitz.ingest.config.schema import IngestConfig

    return load_config(path, schema=IngestConfig, config_type="ingest")


def load_clara_config(path: Optional[Union[str, Path]] = None):
    """
    Load CLaRa configuration.

    Convenience wrapper for load_config(path, config_type="clara").
    """
    # Lazy import to avoid architecture violation
    from fitz.engines.clara.config.schema import ClaraConfig

    return load_config(path, schema=ClaraConfig, config_type="clara")


# =============================================================================
# Config Writing (for fitz init)
# =============================================================================


def save_config(
    data: Union[Dict[str, Any], Any],
    path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Save configuration to a YAML file.

    Args:
        data: Config data (dict or object with to_dict/model_dump)
        path: Path to save to. If None, uses FitzPaths.config()

    Returns:
        Path where config was saved
    """
    if path is None:
        path = FitzPaths.config()

    p = Path(path)

    # Ensure parent directory exists
    p.parent.mkdir(parents=True, exist_ok=True)

    # Convert to dict if needed
    if hasattr(data, "model_dump"):
        data = data.model_dump()
    elif hasattr(data, "to_dict"):
        data = data.to_dict()
    elif not isinstance(data, dict):
        raise ConfigError(f"Cannot save config of type {type(data)}")

    # Write YAML
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved config to {p}")
    return p

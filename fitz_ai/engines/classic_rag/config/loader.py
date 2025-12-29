# fitz_ai/engines/classic_rag/config/loader.py
"""
Configuration loader for Classic RAG engine.

This is the SINGLE loader for Classic RAG configuration.
The old loaders in pipeline/config/ have been consolidated here.

Usage:
    >>> from fitz_ai.engines.classic_rag.config.loader import load_config
    >>> config = load_config()  # Loads user config or default.yaml
    >>> config = load_config("my_config.yaml")  # Loads custom config
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from fitz_ai.engines.classic_rag.config.schema import ClassicRagConfig

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _load_yaml(path: Path) -> dict:
    """Load YAML file and return dict."""
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    # Handle nested classic_rag: key for unified config format
    if "classic_rag" in data and isinstance(data["classic_rag"], dict):
        return data["classic_rag"]

    return data


def get_default_config_path() -> Path:
    """Get path to the package default Classic RAG config file."""
    return DEFAULT_CONFIG_PATH


def get_user_config_path() -> Path:
    """Get path to user's Classic RAG config file (.fitz/config/classic_rag.yaml)."""
    from fitz_ai.core.paths import FitzPaths

    return FitzPaths.engine_config("classic_rag")


def load_config(path: Optional[str] = None) -> ClassicRagConfig:
    """
    Load Classic RAG configuration.

    Resolution order:
    1. Explicit path if provided
    2. User config at .fitz/config/classic_rag.yaml
    3. Package default at fitz_ai/engines/classic_rag/config/default.yaml

    Args:
        path: Optional path to YAML config file.
              If None, uses resolution order.

    Returns:
        Validated ClassicRagConfig instance

    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config path points to a directory
        ValidationError: If config doesn't match schema

    Examples:
        Load default config:
        >>> config = load_config()

        Load custom config:
        >>> config = load_config("my_config.yaml")
    """
    if path is None:
        # Check for user config first
        user_config = get_user_config_path()
        if user_config.exists():
            config_path = user_config
        else:
            # Fall back to package default
            config_path = get_default_config_path()
    else:
        config_path = Path(path)
        if config_path.is_dir():
            raise ValueError(f"Config path points to a directory: {config_path}")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    raw = _load_yaml(config_path)
    return ClassicRagConfig.from_dict(raw)


def load_config_dict(path: Optional[str] = None) -> dict:
    """
    Load configuration as raw dictionary (for advanced use cases).

    Most users should use load_config() instead.

    Resolution order:
    1. Explicit path if provided
    2. User config at .fitz/config/classic_rag.yaml
    3. Package default at fitz_ai/engines/classic_rag/config/default.yaml

    Args:
        path: Optional path to YAML config file.

    Returns:
        Raw configuration dictionary
    """
    if path is None:
        # Check for user config first
        user_config = get_user_config_path()
        if user_config.exists():
            config_path = user_config
        else:
            # Fall back to package default
            config_path = get_default_config_path()
    else:
        config_path = Path(path)

    return _load_yaml(config_path)

# fitz_ai/engines/classic_rag/config/loader.py
"""
Configuration loader for Classic RAG engine.

This is the SINGLE loader for Classic RAG configuration.
The old loaders in pipeline/config/ have been consolidated here.

Usage:
    >>> from fitz_ai.engines.classic_rag.config.loader import load_config
    >>> config = load_config()  # Loads default.yaml
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
        return yaml.safe_load(f) or {}


def _default_config_path() -> Path:
    """Get path to default.yaml in this directory."""
    return DEFAULT_CONFIG_PATH


def load_config(path: Optional[str] = None) -> ClassicRagConfig:
    """
    Load Classic RAG configuration.

    Args:
        path: Optional path to YAML config file.
              If None, loads the built-in default.yaml.

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
        config_path = _default_config_path()
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

    Args:
        path: Optional path to YAML config file.

    Returns:
        Raw configuration dictionary
    """
    if path is None:
        config_path = _default_config_path()
    else:
        config_path = Path(path)

    return _load_yaml(config_path)

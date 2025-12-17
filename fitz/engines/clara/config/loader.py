# fitz/engines/clara/config/loader.py
"""
Configuration loader for CLaRa engine.

This is a thin wrapper around fitz.core.config.
All the actual loading logic lives there.

For new code, prefer importing directly from fitz.core.config:
    from fitz.core.config import load_config, load_clara_config
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from fitz.core.config import (
    load_config as _load_config_core,
    ConfigError,
    ConfigNotFoundError,
)
from fitz.engines.clara.config.schema import ClaraConfig


def load_clara_config(config_path: Optional[Union[str, Path]] = None) -> ClaraConfig:
    """
    Load CLaRa configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, returns defaults.

    Returns:
        ClaraConfig object

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    if config_path is None:
        return ClaraConfig()

    try:
        return _load_config_core(config_path, schema=ClaraConfig)
    except ConfigNotFoundError:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    except ConfigError as e:
        raise RuntimeError(f"Failed to load CLaRa config: {e}") from e


__all__ = [
    "load_clara_config",
]
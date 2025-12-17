# fitz/ingest/config/loader.py
"""
Configuration loader for ingestion pipeline.

This is a thin wrapper around fitz.core.config.
All the actual loading logic lives there.

For new code, prefer importing directly from fitz.core.config:
    from fitz.core.config import load_config, load_ingest_config
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from fitz.core.config import (
    ConfigError,
    ConfigNotFoundError,
    ConfigValidationError,
)
from fitz.core.config import load_config as _load_config_core
from fitz.ingest.config.schema import IngestConfig


# Backwards compatibility alias
class IngestConfigError(ConfigError):
    """
    Raised for ingest configuration errors.

    Deprecated: Use fitz.core.config.ConfigError instead.
    """

    pass


def load_ingest_config(path: Union[str, Path]) -> IngestConfig:
    """
    Load ingestion configuration.

    Args:
        path: Path to YAML config file (required for ingest)

    Returns:
        Validated IngestConfig instance

    Raises:
        IngestConfigError: If config file not found or invalid

    Examples:
        >>> config = load_ingest_config("ingest_config.yaml")
    """
    try:
        return _load_config_core(path, schema=IngestConfig)
    except ConfigNotFoundError as e:
        raise IngestConfigError(f"Config file not found: {path}") from e
    except ConfigValidationError as e:
        raise IngestConfigError("Invalid ingest configuration") from e
    except ConfigError as e:
        raise IngestConfigError(f"Failed to load ingest config: {path}") from e


# Re-export for backwards compatibility
__all__ = [
    "load_ingest_config",
    "IngestConfigError",
]

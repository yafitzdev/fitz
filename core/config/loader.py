# core/config/loader.py
"""
Configuration loader for Fitz.

Responsibilities:
- Load default config
- Load user config (optional)
- Expand ${ENV_VAR} placeholders
- Validate via schema
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from core.logging.logger import get_logger
from core.logging.tags import CLI
from core.config.schema import FitzConfig

logger = get_logger(__name__)


DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _expand_env(value: Any) -> Any:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return _expand_env(data)


def load_config(user_config_path: Path | None = None) -> FitzConfig:
    """
    Load and validate Fitz configuration.

    Precedence:
    - defaults
    - user config (overrides defaults)
    """
    logger.debug(f"{CLI} Loading default config from {DEFAULT_CONFIG_PATH}")
    base_cfg = _load_yaml(DEFAULT_CONFIG_PATH)

    if user_config_path:
        logger.debug(f"{CLI} Loading user config from {user_config_path}")
        user_cfg = _load_yaml(user_config_path)
        base_cfg.update(user_cfg)

    return FitzConfig.model_validate(base_cfg)

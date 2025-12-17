# fitz/core/config/loader.py
"""
Configuration loader for Fitz.

Responsibilities:
- Load default config (meta)
- Load user config (meta)
- Expand ${ENV_VAR} placeholders
- Resolve preset
- Normalize preset into runtime config
- Validate resolved config via FitzConfig
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from fitz.engines.classic_rag.config.normalize import normalize_preset
from fitz.engines.classic_rag.config.schema import FitzConfig, FitzMetaConfig
from fitz.logging.logger import get_logger
from fitz.logging.tags import CLI

logger = get_logger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.yaml"


def _expand_env(value: object) -> object:
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


def _deep_merge(a: dict[str, object], b: dict[str, object]) -> dict[str, object]:
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got: {type(data).__name__}")

    return _expand_env(data)


def load_config(user_config_path: Path | None = None) -> FitzConfig:
    """
    Load and validate Fitz runtime configuration.

    Flow:
    1. Load meta config
    2. Merge user meta config
    3. Resolve preset
    4. Normalize preset into runtime shape
    5. Validate runtime config
    """
    logger.debug(f"{CLI} Loading default config from {DEFAULT_CONFIG_PATH}")
    meta_raw = _load_yaml(DEFAULT_CONFIG_PATH)

    if user_config_path:
        logger.debug(f"{CLI} Loading user config from {user_config_path}")
        user_raw = _load_yaml(user_config_path)
        meta_raw = _deep_merge(meta_raw, user_raw)

    meta = FitzMetaConfig.model_validate(meta_raw)

    preset_name = meta.default_preset
    if preset_name not in meta.presets:
        raise ValueError(f"Unknown preset: {preset_name}")

    preset = meta.presets[preset_name]
    normalized = normalize_preset(preset)

    return FitzConfig.model_validate(normalized)

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    from importlib.resources import files as pkg_files  # Python 3.9+
except ImportError:
    from importlib_resources import files as pkg_files  # type: ignore

from rag.exceptions.config import ConfigError


# Global cache for the resolved config
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    try:
        merged: Dict[str, Any] = dict(base)

        for key, override_val in override.items():
            if key in merged:
                base_val = merged[key]
                if isinstance(base_val, dict) and isinstance(override_val, dict):
                    merged[key] = _deep_merge(base_val, override_val)
                else:
                    merged[key] = override_val
            else:
                merged[key] = override_val

        return merged
    except Exception as e:
        raise ConfigError("Failed to merge configuration values") from e


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as e:
        raise ConfigError(f"Failed to read YAML config: {path}") from e

    if not isinstance(data, dict):
        raise ConfigError(f"Top-level YAML must be a mapping (dict): {path}")

    return data


def _load_default_config() -> Dict[str, Any]:
    """
    Load the default config YAML bundled with the package.
    """
    try:
        default_path = pkg_files("fitz_rag.config").joinpath("default.yaml")
        default_path = Path(default_path)
        return _load_yaml_file(default_path)
    except Exception as e:
        raise ConfigError("Failed to load default.yaml") from e


def _find_user_config_path(explicit_path: Optional[os.PathLike] = None) -> Optional[Path]:
    try:
        if explicit_path is not None:
            return Path(explicit_path)

        env_path = os.environ.get("FITZ_RAG_CONFIG")
        if env_path:
            return Path(env_path)

        return None
    except Exception as e:
        raise ConfigError("Failed to resolve user config path") from e


def load_config(user_config_path: Optional[os.PathLike] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load + merge default and optional user config with defensive exception wrapping.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload and user_config_path is None:
        return _CONFIG_CACHE

    try:
        default_cfg = _load_default_config()
    except Exception as e:
        raise ConfigError("Failed loading default configuration") from e

    # Locate a user config, if any
    try:
        user_path = _find_user_config_path(user_config_path)
    except Exception as e:
        raise ConfigError("Failed determining user configuration path") from e

    # Merge results
    try:
        if user_path is not None and user_path.exists():
            user_cfg = _load_yaml_file(user_path)
            merged = _deep_merge(default_cfg, user_cfg)
        else:
            merged = default_cfg
    except Exception as e:
        raise ConfigError("Failed merging user configuration") from e

    if user_config_path is None and not force_reload:
        _CONFIG_CACHE = merged

    return merged


def get_config() -> Dict[str, Any]:
    """Convenience helper."""
    try:
        return load_config()
    except Exception as e:
        raise ConfigError("Failed retrieving application configuration") from e

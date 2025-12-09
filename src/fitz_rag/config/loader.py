from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

try:
    # Python 3.9+ importlib.resources API
    from importlib.resources import files as pkg_files
except ImportError:
    # Python < 3.9 fallback
    from importlib_resources import files as pkg_files  # type: ignore


# Global cache for the resolved config
_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge `override` into `base`.
    - Dict values are merged recursively.
    - Non-dict values from `override` replace those in `base`.
    Returns a new dict; does not mutate inputs.
    """
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


def _load_yaml_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping at the top level: {path}")
    return data


def _load_default_config() -> Dict[str, Any]:
    """
    Load the default config YAML bundled with the package.
    """
    default_path = pkg_files("fitz_rag.config").joinpath("default.yaml")
    # importlib.resources returns an AbstractPath-like object; cast to Path
    default_path = Path(default_path)
    return _load_yaml_file(default_path)


def _find_user_config_path(explicit_path: Optional[os.PathLike] = None) -> Optional[Path]:
    """
    Resolve the user config path.
    Priority:
      1. explicit path argument
      2. FITZ_RAG_CONFIG environment variable
      3. None (no user config)
    """
    if explicit_path is not None:
        return Path(explicit_path)

    env_path = os.environ.get("FITZ_RAG_CONFIG")
    if env_path:
        return Path(env_path)

    return None


def load_config(user_config_path: Optional[os.PathLike] = None, force_reload: bool = False) -> Dict[str, Any]:
    """
    Load and merge the configuration.

    - Loads default config from the package (default.yaml).
    - Loads optional user config from:
        * `user_config_path` argument, or
        * `FITZ_RAG_CONFIG` environment variable.
    - Deep merges user config over defaults.

    Parameters
    ----------
    user_config_path:
        Optional path to a user YAML config file.
    force_reload:
        If True, ignore cached config and reload from disk.

    Returns
    -------
    dict
        The merged configuration.
    """
    global _CONFIG_CACHE

    if _CONFIG_CACHE is not None and not force_reload and user_config_path is None:
        return _CONFIG_CACHE

    default_cfg = _load_default_config()

    user_path = _find_user_config_path(user_config_path)
    if user_path is not None and user_path.exists():
        user_cfg = _load_yaml_file(user_path)
        merged = _deep_merge(default_cfg, user_cfg)
    else:
        merged = default_cfg

    # Cache only if no explicit user path was passed
    if user_config_path is None and not force_reload:
        _CONFIG_CACHE = merged

    return merged


def get_config() -> Dict[str, Any]:
    """
    Convenience accessor for the app-wide configuration.
    Uses the FITZ_RAG_CONFIG environment variable (if set).
    """
    return load_config()

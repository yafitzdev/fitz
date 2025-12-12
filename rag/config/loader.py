from __future__ import annotations

import os
import yaml
from functools import lru_cache
from typing import Any, Dict
from importlib.resources import files as pkg_files

from rag.config.schema import (
    EmbeddingConfig,
    RetrieverConfig,
    RerankConfig,
    RGSSettings,
    LoggingConfig,
)


class ConfigError(RuntimeError):
    pass


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(a)
    for key, value in b.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        raise ConfigError(f"Failed to load config file: {path}") from exc


def _load_default_config() -> Dict[str, Any]:
    try:
        default_path = pkg_files("rag.config").joinpath("default.yaml")
        return yaml.safe_load(default_path.read_text()) or {}
    except Exception as exc:
        raise ConfigError("Failed to load bundled rag default config") from exc


@lru_cache
def load_config(path: str | None = None) -> Dict[str, Any]:
    config = _load_default_config()

    env_path = os.getenv("FITZ_RAG_CONFIG")
    override_path = path or env_path

    if override_path:
        override = _load_yaml(override_path)
        config = _deep_merge(config, override)

    return config


def get_config() -> Dict[str, Any]:
    raw = load_config()

    try:
        return {
            "embedding": EmbeddingConfig(**raw.get("embedding", {})),
            "retrieval": RetrieverConfig(**raw.get("retrieval", {})),
            "rerank": RerankConfig(**raw.get("rerank", {})),
            "rgs": RGSSettings(**raw.get("rgs", {})),
            "logging": LoggingConfig(**raw.get("logging", {})),
        }
    except Exception as exc:
        raise ConfigError("Invalid rag configuration") from exc

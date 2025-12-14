from __future__ import annotations

from pathlib import Path
from typing import Optional

import yaml

from fitz.pipeline.config.schema import RAGConfig


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _default_config_path() -> Path:
    # pipeline/config/default.yaml
    return Path(__file__).parent / "default.yaml"


def load_config(path: Optional[str]) -> dict:
    """
    Load RAG config.

    - None → load built-in default.yaml
    - str  → load provided yaml file
    """
    if path is None:
        return _load_yaml(_default_config_path())

    p = Path(path)

    if p.is_dir():
        raise ValueError(f"Config path points to a directory: {p}")

    return _load_yaml(p)

from __future__ import annotations

from pathlib import Path
import yaml

from ingest.config.schema import IngestConfig


def load_ingest_config(path: str | Path) -> IngestConfig:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return IngestConfig(**data)

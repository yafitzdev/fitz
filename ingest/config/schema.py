# ingest/config/schema.py
from __future__ import annotations

from typing import Optional, Dict, Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------
# Sub-configs (internal, structured, explicit)
# ---------------------------------------------------------
class IngesterConfig(BaseModel):
    plugin_name: str
    options: Dict[str, Any] = Field(default_factory=dict)


class ChunkerConfig(BaseModel):
    plugin_name: str
    chunk_size: int = 1000
    chunk_overlap: int = 0
    options: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------
# Top-level ingestion pipeline config
# ---------------------------------------------------------
class IngestConfig(BaseModel):
    """
    Central configuration for the ENTIRE ingestion pipeline.

    This config intentionally spans:
    - ingester selection
    - chunker selection
    - target collection

    Engines MUST NOT own config.
    Engines are BUILT FROM this config.
    """

    ingester: IngesterConfig
    chunker: ChunkerConfig
    collection: str

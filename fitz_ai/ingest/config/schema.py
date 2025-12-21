# fitz_ai/ingest/config/schema.py

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class IngesterConfig(BaseModel):
    plugin_name: str = Field(..., description="Ingester plugin name")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Ingester init kwargs"
    )

    model_config = ConfigDict(extra="forbid")


class ChunkerConfig(BaseModel):
    """
    Plugin-agnostic chunker configuration.

    All chunker-specific parameters go in kwargs.
    No hardcoded chunk_size or chunk_overlap - plugins define their own params.

    Examples:
        Simple chunker:
        >>> ChunkerConfig(
        ...     plugin_name="simple",
        ...     kwargs={"chunk_size": 1000}
        ... )

        Overlap chunker:
        >>> ChunkerConfig(
        ...     plugin_name="overlap",
        ...     kwargs={"chunk_size": 1000, "chunk_overlap": 200}
        ... )
    """

    plugin_name: str = Field(..., description="Chunker plugin name")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="All chunker parameters"
    )

    model_config = ConfigDict(extra="forbid")


class IngestConfig(BaseModel):
    """
    Central configuration for the entire ingestion pipeline.

    Rules:
    - Engines are built FROM config (engines do not own config).
    - Provider/plugin selection lives only here (plugin_name).
    """

    ingester: IngesterConfig
    chunker: ChunkerConfig
    collection: str = Field(..., description="Target vector DB collection")

    model_config = ConfigDict(extra="forbid")

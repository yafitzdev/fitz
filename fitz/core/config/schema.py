# fitz/core/config/schema.py
"""
Pydantic schema for Fitz configuration.

Architecture:
- FitzMetaConfig: YAML-facing meta config (presets, defaults)
- FitzConfig: fully resolved runtime config consumed by engines
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PluginConfig(BaseModel):
    """
    Generic plugin configuration block.

    Provider-specific configuration must be expressed via:
    - plugin_name: plugin id in the central registry
    - kwargs: arbitrary plugin init kwargs
    """

    plugin_name: str = Field(..., description="Plugin name in the central registry")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin init kwargs")

    model_config = ConfigDict(extra="forbid")


class FitzConfig(BaseModel):
    """
    Fully resolved runtime configuration.

    This is the ONLY config that engines are allowed to see.
    """

    chat: PluginConfig
    embedding: PluginConfig
    rerank: PluginConfig | None = Field(default=None, description="Optional rerank plugin")

    vector_db: PluginConfig
    pipeline: PluginConfig

    model_config = ConfigDict(extra="forbid")


class FitzMetaConfig(BaseModel):
    """
    YAML-facing meta configuration.

    This layer is NEVER passed to engines.
    """

    default_preset: str
    presets: dict[str, dict[str, Any]]

    model_config = ConfigDict(extra="forbid")

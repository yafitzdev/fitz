# fitz_sage/core/config.py
"""
Shared plugin config base.

Paradigm-agnostic config types only — no engine imports. Engine-specific config
loading lives in each engine's package; runtime-layer loading (merge of package
defaults with user overrides) lives in ``fitz_sage.config.loader``.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class BasePluginConfig(BaseModel):
    """Base class to avoid repeating model_config."""

    model_config = ConfigDict(extra="forbid")

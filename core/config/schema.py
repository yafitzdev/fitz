# core/config/schema.py
"""
Pydantic schema for Fitz configuration.

Rules:
- Strict validation
- No unknown keys
- Single top-level llm block
"""

from __future__ import annotations

from pydantic import BaseModel, Field, ConfigDict


class LLMConfig(BaseModel):
    provider: str = Field(..., description="LLM provider name (e.g. openai, anthropic)")
    model: str = Field(..., description="Model identifier")
    api_key: str | None = Field(
        default=None,
        description="Optional explicit API key (env vars preferred)",
    )

    model_config = ConfigDict(extra="forbid")


class FitzConfig(BaseModel):
    llm: LLMConfig

    model_config = ConfigDict(extra="forbid")

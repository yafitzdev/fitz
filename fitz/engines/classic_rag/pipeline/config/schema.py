# pipeline/config/schema.py
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PipelinePluginConfig(BaseModel):
    plugin_name: str = Field(..., description="Plugin name in the central registry")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin init kwargs")
    model_config = ConfigDict(extra="forbid")


class RerankConfig(BaseModel):
    enabled: bool = False
    plugin_name: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)
    model_config = ConfigDict(extra="forbid")


class RetrieverConfig(BaseModel):
    plugin_name: str = "dense"
    collection: str
    top_k: int = 5
    model_config = ConfigDict(extra="forbid")


class RGSConfig(BaseModel):
    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: str | None = None
    max_chunks: int = 8
    max_answer_chars: int | None = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"
    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    level: str = "INFO"
    model_config = ConfigDict(extra="forbid")


class RAGConfig(BaseModel):
    vector_db: PipelinePluginConfig
    llm: PipelinePluginConfig
    embedding: PipelinePluginConfig
    retriever: RetrieverConfig
    rerank: RerankConfig = Field(default_factory=RerankConfig)
    rgs: RGSConfig = Field(default_factory=RGSConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, data: dict) -> "RAGConfig":
        """
        Create RAGConfig from a dictionary.

        Uses Pydantic's validation to create a config instance from a dict.
        This is useful for creating configs from presets or API payloads.

        Args:
            data: Configuration dictionary

        Returns:
            Validated RAGConfig instance

        Example:
            >>> from fitz.engines.classic_rag.config.presets import get_preset
            >>> config_dict = get_preset("local")
            >>> config = RAGConfig.from_dict(config_dict)
        """
        return cls.model_validate(data)

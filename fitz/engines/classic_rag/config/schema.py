# fitz/engines/classic_rag/config/schema.py
"""
Configuration schema for Classic RAG engine.

This is the SINGLE source of truth for Classic RAG configuration.
All other config schemas have been consolidated here.

Schema hierarchy:
- ClassicRagConfig: The main config consumed by the engine
- PluginConfig: Generic plugin configuration block
- RetrieverConfig: Retriever-specific settings
- RerankConfig: Reranker settings
- RGSConfig: Retrieval-guided synthesis settings
- LoggingConfig: Logging settings
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class PluginConfig(BaseModel):
    """
    Generic plugin configuration block.

    Used for chat, embedding, vector_db, and other pluggable components.

    Examples:
        >>> config = PluginConfig(
        ...     plugin_name="openai",
        ...     kwargs={"model": "gpt-4", "temperature": 0.2}
        ... )
    """

    plugin_name: str = Field(..., description="Plugin name in the central registry")
    kwargs: dict[str, Any] = Field(default_factory=dict, description="Plugin init kwargs")

    model_config = ConfigDict(extra="forbid")


class RerankConfig(BaseModel):
    """
    Reranker configuration.

    Reranking is optional but can significantly improve retrieval quality.
    """

    enabled: bool = False
    plugin_name: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


class RetrieverConfig(BaseModel):
    """
    Retriever configuration.

    Defines how chunks are retrieved from the vector database.
    """

    plugin_name: str = "dense"
    collection: str = Field(..., description="Vector DB collection name")
    top_k: int = Field(default=5, ge=1, description="Number of chunks to retrieve")

    model_config = ConfigDict(extra="forbid")


class RGSConfig(BaseModel):
    """
    Retrieval-Guided Synthesis configuration.

    Controls how the LLM generates answers from retrieved context.
    """

    enable_citations: bool = True
    strict_grounding: bool = True
    answer_style: str | None = None
    max_chunks: int = Field(default=8, ge=1)
    max_answer_chars: int | None = None
    include_query_in_context: bool = True
    source_label_prefix: str = "S"

    model_config = ConfigDict(extra="forbid")


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"

    model_config = ConfigDict(extra="forbid")


class ClassicRagConfig(BaseModel):
    """
    Complete configuration for Classic RAG engine.

    This is the ONLY config class that the Classic RAG engine consumes.
    It contains all settings needed to build and run the RAG pipeline.

    Examples:
        Load from YAML:
        >>> from fitz.engines.classic_rag.config.loader import load_config
        >>> config = load_config("config.yaml")

        Create from dict:
        >>> config = ClassicRagConfig.from_dict({
        ...     "chat": {"plugin_name": "openai", "kwargs": {"model": "gpt-4"}},
        ...     "embedding": {"plugin_name": "openai", "kwargs": {}},
        ...     "vector_db": {"plugin_name": "qdrant", "kwargs": {}},
        ...     "retriever": {"plugin_name": "dense", "collection": "docs", "top_k": 5},
        ... })
    """

    # Required plugins - 'chat' is the canonical name (not 'llm')
    chat: PluginConfig = Field(..., description="Chat/LLM plugin configuration")
    embedding: PluginConfig = Field(..., description="Embedding plugin configuration")
    vector_db: PluginConfig = Field(..., description="Vector database plugin configuration")

    # Retriever (required)
    retriever: RetrieverConfig = Field(..., description="Retriever configuration")

    # Optional components
    rerank: RerankConfig = Field(default_factory=RerankConfig, description="Reranker configuration")
    rgs: RGSConfig = Field(default_factory=RGSConfig, description="RGS configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, data: dict) -> "ClassicRagConfig":
        """
        Create config from a dictionary.

        Uses Pydantic's validation to create a config instance.

        Args:
            data: Configuration dictionary

        Returns:
            Validated ClassicRagConfig instance
        """
        return cls.model_validate(data)


# =============================================================================
# BACKWARDS COMPATIBILITY ALIASES
# =============================================================================
# These aliases allow existing code to continue working during migration.
# TODO: Remove after all imports are updated.

RAGConfig = ClassicRagConfig
PipelinePluginConfig = PluginConfig
EnginePluginConfig = PluginConfig

# FitzConfig was the old name - alias to ClassicRagConfig
FitzConfig = ClassicRagConfig
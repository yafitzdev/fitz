# fitz_ai/engines/classic_rag/config/schema.py
"""
Configuration schema for Classic RAG engine.

This is the SINGLE source of truth for Classic RAG configuration.

Schema hierarchy:
- ClassicRagConfig: The main config consumed by the engine
- PluginConfig: Generic plugin configuration block
- RetrievalConfig: Step-based retrieval configuration
- RerankConfig: Reranker settings
- RGSConfig: Retrieval-guided synthesis settings
- LoggingConfig: Logging settings
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# =============================================================================
# Plugin Configuration
# =============================================================================


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
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Plugin init kwargs"
    )

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Retrieval Configuration
# =============================================================================


class RetrievalConfig(BaseModel):
    """
    Retrieval configuration.

    References a YAML-based retrieval plugin that defines the step pipeline.
    Plugin files live in: fitz_ai/engines/classic_rag/retrieval/runtime/plugins/*.yaml

    Example YAML:
    ```yaml
    retrieval:
      plugin_name: dense        # References dense.yaml
      collection: my_docs
      top_k: 5
    ```

    Available plugins:
    - dense: Standard vector search with optional reranking
    - dense_rerank: Vector search + rerank + threshold + dedupe
    """

    plugin_name: str = Field(default="dense", description="Retrieval plugin name (references YAML file)")
    collection: str = Field(..., description="Vector DB collection name")
    top_k: int = Field(default=5, ge=1, description="Final number of chunks to return")

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Rerank Configuration
# =============================================================================


class RerankConfig(BaseModel):
    """
    Reranker configuration.

    Reranking is optional but can significantly improve retrieval quality.
    When enabled, provides the reranker dependency for rerank steps.
    """

    enabled: bool = False
    plugin_name: str | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# RGS Configuration
# =============================================================================


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


# =============================================================================
# Logging Configuration
# =============================================================================


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = "INFO"

    model_config = ConfigDict(extra="forbid")


# =============================================================================
# Main Configuration
# =============================================================================


class ClassicRagConfig(BaseModel):
    """
    Complete configuration for Classic RAG engine.

    This is the ONLY config class that the Classic RAG engine consumes.
    It contains all settings needed to build and run the RAG pipeline.

    Examples:
        Load from YAML:
        >>> from fitz_ai.engines.classic_rag.config import load_config
        >>> config = load_config("config.yaml")

        Create from dict:
        >>> config = ClassicRagConfig.from_dict({
        ...     "chat": {"plugin_name": "openai", "kwargs": {"model": "gpt-4"}},
        ...     "embedding": {"plugin_name": "openai", "kwargs": {}},
        ...     "vector_db": {"plugin_name": "qdrant", "kwargs": {}},
        ...     "retrieval": {"collection": "docs", "top_k": 5},
        ... })

        With explicit steps:
        >>> config = ClassicRagConfig.from_dict({
        ...     "chat": {"plugin_name": "cohere", "kwargs": {}},
        ...     "embedding": {"plugin_name": "cohere", "kwargs": {}},
        ...     "vector_db": {"plugin_name": "qdrant", "kwargs": {}},
        ...     "retrieval": {
        ...         "collection": "docs",
        ...         "steps": [
        ...             {"type": "vector_search", "k": 25},
        ...             {"type": "rerank", "k": 10},
        ...             {"type": "threshold", "threshold": 0.5},
        ...             {"type": "limit", "k": 5},
        ...         ]
        ...     },
        ...     "rerank": {"enabled": True, "plugin_name": "cohere"},
        ... })
    """

    # Required plugins
    chat: PluginConfig = Field(..., description="Chat/LLM plugin configuration")
    embedding: PluginConfig = Field(..., description="Embedding plugin configuration")
    vector_db: PluginConfig = Field(..., description="Vector database plugin configuration")

    # Retrieval (new step-based config)
    retrieval: RetrievalConfig = Field(..., description="Retrieval pipeline configuration")

    # Optional components
    rerank: RerankConfig = Field(
        default_factory=RerankConfig, description="Reranker configuration"
    )
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
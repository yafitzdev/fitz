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

from pydantic import BaseModel, ConfigDict, Field, field_validator

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
# Ingestion Configuration
# =============================================================================


class IngesterConfig(BaseModel):
    """Configuration for the ingestion plugin."""

    plugin_name: str = Field(..., description="Ingester plugin name")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Ingester init kwargs"
    )

    model_config = ConfigDict(extra="forbid")


class ExtensionChunkerConfig(BaseModel):
    """
    Configuration for a chunker assigned to a specific file extension.

    Example:
        >>> ExtensionChunkerConfig(
        ...     plugin_name="markdown",
        ...     kwargs={"max_tokens": 800, "preserve_headers": True}
        ... )
    """

    plugin_name: str = Field(..., description="Chunker plugin name")
    kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Chunker-specific parameters"
    )

    model_config = ConfigDict(extra="forbid")


class ChunkingRouterConfig(BaseModel):
    """
    Configuration for the ChunkingRouter.

    Supports file-type specific chunking with a default fallback.

    Attributes:
        default: Default chunker for extensions not in by_extension.
        by_extension: Mapping of file extensions to chunker configs.
        warn_on_fallback: Whether to log warnings when using default chunker.

    Example YAML:
        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
              chunk_overlap: 0

          by_extension:
            .md:
              plugin_name: markdown
              kwargs:
                max_tokens: 800
            .py:
              plugin_name: python_code
              kwargs:
                chunk_by: function
            .pdf:
              plugin_name: pdf_sections
              kwargs:
                max_section_chars: 2000

          warn_on_fallback: true
    """

    default: ExtensionChunkerConfig = Field(
        ..., description="Default chunker for unknown extensions"
    )
    by_extension: dict[str, ExtensionChunkerConfig] = Field(
        default_factory=dict,
        description="Mapping of extensions to chunker configs",
    )
    warn_on_fallback: bool = Field(
        default=True,
        description="Log warning when using default chunker for unknown extension",
    )

    model_config = ConfigDict(extra="forbid")

    @field_validator("by_extension", mode="before")
    @classmethod
    def normalize_extensions(cls, v: dict[str, Any]) -> dict[str, Any]:
        """Ensure all extension keys start with a dot and are lowercase."""
        if not isinstance(v, dict):
            return v

        normalized = {}
        for ext, config in v.items():
            norm_ext = ext.lower()
            if not norm_ext.startswith("."):
                norm_ext = f".{norm_ext}"
            normalized[norm_ext] = config

        return normalized


class IngestConfig(BaseModel):
    """
    Configuration for the ingestion pipeline.

    Example YAML:
        ingest:
          ingester:
            plugin_name: local
            kwargs: {}

          chunking:
            default:
              plugin_name: simple
              kwargs:
                chunk_size: 1000
            by_extension:
              .md:
                plugin_name: markdown
                kwargs:
                  max_tokens: 800

          collection: my_docs
    """

    ingester: IngesterConfig = Field(..., description="Ingestion plugin config")
    chunking: ChunkingRouterConfig = Field(..., description="Chunking router config")
    collection: str = Field(..., description="Target vector DB collection")

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

    plugin_name: str = Field(
        default="dense", description="Retrieval plugin name (references YAML file)"
    )
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
    vector_db: PluginConfig = Field(
        ..., description="Vector database plugin configuration"
    )

    # Retrieval (new step-based config)
    retrieval: RetrievalConfig = Field(
        ..., description="Retrieval pipeline configuration"
    )

    # Optional components
    rerank: RerankConfig = Field(
        default_factory=RerankConfig, description="Reranker configuration"
    )
    rgs: RGSConfig = Field(default_factory=RGSConfig, description="RGS configuration")
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    # Ingestion (optional - only needed when ingesting documents)
    ingest: IngestConfig | None = Field(
        default=None, description="Ingestion pipeline configuration"
    )

    # Chunking (optional - can be at top level for convenience, used by fitz ingest)
    chunking: ChunkingRouterConfig | None = Field(
        default=None,
        description="Chunking configuration (top-level for CLI convenience)",
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

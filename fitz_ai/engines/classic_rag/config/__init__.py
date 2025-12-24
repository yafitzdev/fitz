# fitz_ai/engines/classic_rag/config/__init__.py
"""
Configuration for Classic RAG engine.

This package provides:
- ClassicRagConfig: The main configuration schema
- load_config: Configuration loader function
- Supporting schemas (PluginConfig, RetrievalConfig, IngestConfig, etc.)

Usage:
    >>> from fitz_ai.engines.classic_rag.config import load_config, ClassicRagConfig
    >>> config = load_config("config.yaml")
"""

from .loader import (
    DEFAULT_CONFIG_PATH,
    load_config,
    load_config_dict,
)
from .schema import (  # Main config; RAG sub-configs; Ingestion configs
    ChunkingRouterConfig,
    ClassicRagConfig,
    ExtensionChunkerConfig,
    IngestConfig,
    IngesterConfig,
    LoggingConfig,
    PluginConfig,
    RerankConfig,
    RetrievalConfig,
    RGSConfig,
)

__all__ = [
    # Main config
    "ClassicRagConfig",
    "load_config",
    "load_config_dict",
    # RAG sub-configs
    "PluginConfig",
    "RetrievalConfig",
    "RerankConfig",
    "RGSConfig",
    "LoggingConfig",
    # Ingestion configs
    "IngestConfig",
    "IngesterConfig",
    "ChunkingRouterConfig",
    "ExtensionChunkerConfig",
    # Constants
    "DEFAULT_CONFIG_PATH",
]

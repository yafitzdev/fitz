# fitz/engines/classic_rag/config/__init__.py
"""
Configuration for Classic RAG engine.

This package provides:
- ClassicRagConfig: The main configuration schema
- load_config: Configuration loader function
- Supporting schemas (PluginConfig, RetrieverConfig, etc.)

Usage:
    >>> from fitz_ai.engines.classic_rag.config import load_config, ClassicRagConfig
    >>> config = load_config("config.yaml")
"""

from .loader import (  # Backwards compatibility exports
    DEFAULT_CONFIG_PATH,
    _load_yaml,
    load_config,
    load_config_dict,
)
from .schema import (  # Main config; Sub-configs; Backwards compatibility aliases
    ClassicRagConfig,
    EnginePluginConfig,
    FitzConfig,
    LoggingConfig,
    PipelinePluginConfig,
    PluginConfig,
    RAGConfig,
    RerankConfig,
    RetrieverConfig,
    RGSConfig,
)

__all__ = [
    # Main config
    "ClassicRagConfig",
    "load_config",
    "load_config_dict",
    # Sub-configs
    "PluginConfig",
    "RetrieverConfig",
    "RerankConfig",
    "RGSConfig",
    "LoggingConfig",
    # Backwards compatibility
    "RAGConfig",
    "PipelinePluginConfig",
    "EnginePluginConfig",
    "FitzConfig",
    "DEFAULT_CONFIG_PATH",
    "_load_yaml",
]

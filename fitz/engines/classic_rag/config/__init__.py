# fitz/engines/classic_rag/config/__init__.py
"""
Configuration for Classic RAG engine.

This package provides:
- ClassicRagConfig: The main configuration schema
- load_config: Configuration loader function
- Supporting schemas (PluginConfig, RetrieverConfig, etc.)

Usage:
    >>> from fitz.engines.classic_rag.config import load_config, ClassicRagConfig
    >>> config = load_config("config.yaml")
"""

from .schema import (
    # Main config
    ClassicRagConfig,
    # Sub-configs
    PluginConfig,
    RetrieverConfig,
    RerankConfig,
    RGSConfig,
    LoggingConfig,
    # Backwards compatibility aliases
    RAGConfig,
    PipelinePluginConfig,
    EnginePluginConfig,
    FitzConfig,
)

from .loader import (
    load_config,
    load_config_dict,
    # Backwards compatibility exports
    DEFAULT_CONFIG_PATH,
    _load_yaml,
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

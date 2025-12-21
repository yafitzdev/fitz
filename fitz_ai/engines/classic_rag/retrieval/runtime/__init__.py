# fitz_ai/engines/classic_rag/retrieval/runtime/__init__.py
"""
Retrieval Runtime - YAML-based plugin system.

The retrieval system uses YAML files to define retrieval pipelines
that orchestrate standard Python step classes.

Usage:
    from fitz_ai.engines.classic_rag.retrieval.runtime import (
        get_retrieval_plugin,
        available_retrieval_plugins,
    )

    pipeline = get_retrieval_plugin(
        plugin_name="dense",
        vector_client=my_client,
        embedder=my_embedder,
        collection="my_docs",
    )

    chunks = pipeline.retrieve("my query")
"""

from .loader import (
    RetrievalDependencies,
    RetrievalPipelineFromYaml,
    RetrievalPluginSpec,
    create_retrieval_pipeline,
    list_available_plugins,
    load_plugin_spec,
)
from .registry import (
    PluginNotFoundError,
    available_retrieval_plugins,
    get_retrieval_plugin,
)

__all__ = [
    # Main API
    "get_retrieval_plugin",
    "available_retrieval_plugins",
    "create_retrieval_pipeline",
    # Types
    "RetrievalPipelineFromYaml",
    "RetrievalDependencies",
    "RetrievalPluginSpec",
    # Utilities
    "list_available_plugins",
    "load_plugin_spec",
    # Errors
    "PluginNotFoundError",
]
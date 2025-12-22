# fitz_ai/engines/classic_rag/retrieval/registry.py
"""
Retrieval Plugin Registry (YAML-based).

Discovers and loads retrieval plugins from YAML files in the plugins directory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fitz_ai.engines.classic_rag.retrieval.loader import (
    RetrievalDependencies,
    RetrievalPipelineFromYaml,
    create_retrieval_pipeline,
    list_available_plugins,
    load_plugin_spec,
)

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.retrieval.steps import (
        Embedder,
        Reranker,
        VectorClient,
    )


class PluginNotFoundError(Exception):
    """Raised when requested plugin doesn't exist."""

    pass


def get_retrieval_plugin(
    plugin_name: str,
    vector_client: "VectorClient",
    embedder: "Embedder",
    collection: str,
    reranker: "Reranker | None" = None,
    top_k: int = 5,
) -> RetrievalPipelineFromYaml:
    """
    Get a retrieval plugin by name.

    Args:
        plugin_name: Name of the plugin (e.g., "dense", "dense_rerank")
        vector_client: Vector database client
        embedder: Embedding service
        collection: Collection name
        reranker: Optional reranking service
        top_k: Final number of chunks to return

    Returns:
        Configured retrieval pipeline

    Raises:
        PluginNotFoundError: If plugin doesn't exist
    """
    try:
        return create_retrieval_pipeline(
            plugin_name=plugin_name,
            vector_client=vector_client,
            embedder=embedder,
            collection=collection,
            reranker=reranker,
            top_k=top_k,
        )
    except FileNotFoundError as e:
        raise PluginNotFoundError(str(e)) from e


def available_retrieval_plugins() -> list[str]:
    """List available retrieval plugins."""
    return list_available_plugins()


__all__ = [
    "get_retrieval_plugin",
    "available_retrieval_plugins",
    "PluginNotFoundError",
    "RetrievalPipelineFromYaml",
    "RetrievalDependencies",
]
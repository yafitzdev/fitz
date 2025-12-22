# fitz_ai/engines/classic_rag/retrieval/__init__.py
"""
Retrieval subsystem for Classic RAG.

Provides YAML-based retrieval plugins with composable step pipelines.

Usage:
    from fitz_ai.engines.classic_rag.retrieval import (
        get_retrieval_plugin,
        available_retrieval_plugins,
    )

    # Or import steps directly
    from fitz_ai.engines.classic_rag.retrieval.steps import (
        VectorSearchStep,
        RerankStep,
        ThresholdStep,
        LimitStep,
        DedupeStep,
    )
"""

# Re-export step classes for convenience
from .steps import (
    STEP_REGISTRY,
    DedupeStep,
    Embedder,
    LimitStep,
    Reranker,
    RerankStep,
    RetrievalStep,
    ThresholdStep,
    VectorClient,
    VectorSearchStep,
    get_step_class,
    list_available_steps,
)

__all__ = [
    # Step classes
    "RetrievalStep",
    "VectorSearchStep",
    "RerankStep",
    "ThresholdStep",
    "LimitStep",
    "DedupeStep",
    # Protocols
    "VectorClient",
    "Embedder",
    "Reranker",
    # Registry
    "STEP_REGISTRY",
    "get_step_class",
    "list_available_steps",
]
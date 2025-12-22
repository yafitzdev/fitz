# fitz_ai/engines/classic_rag/retrieval/steps/__init__.py
"""
Retrieval Steps - Standard composable building blocks.

These steps are always the same Python logic. Users orchestrate them
via YAML config by specifying which steps to run and their parameters.

Pipeline example: vector_search(k=25) → rerank(k=10) → threshold(τ) → limit(k=5)

Each step:
- Takes a query + list of chunks (or nothing for initial step)
- Returns a list of chunks
- Is stateless and reusable

Usage:
    from fitz_ai.engines.classic_rag.retrieval.steps import (
        VectorSearchStep,
        RerankStep,
        ThresholdStep,
        LimitStep,
        DedupeStep,
        get_step_class,
        list_available_steps,
    )
"""

from .base import Embedder, Reranker, RetrievalStep, VectorClient
from .dedupe import DedupeStep
from .limit import LimitStep
from .rerank import RerankStep
from .threshold import ThresholdStep
from .vector_search import VectorSearchStep

# =============================================================================
# Step Registry
# =============================================================================

STEP_REGISTRY: dict[str, type[RetrievalStep]] = {
    "vector_search": VectorSearchStep,
    "rerank": RerankStep,
    "threshold": ThresholdStep,
    "limit": LimitStep,
    "dedupe": DedupeStep,
}


def get_step_class(step_type: str) -> type[RetrievalStep]:
    """Get step class by type name."""
    if step_type not in STEP_REGISTRY:
        available = list(STEP_REGISTRY.keys())
        raise ValueError(f"Unknown step type: {step_type!r}. Available: {available}")
    return STEP_REGISTRY[step_type]


def list_available_steps() -> list[str]:
    """List all available step types."""
    return list(STEP_REGISTRY.keys())


__all__ = [
    # Base classes and protocols
    "RetrievalStep",
    "VectorClient",
    "Embedder",
    "Reranker",
    # Step classes
    "VectorSearchStep",
    "RerankStep",
    "ThresholdStep",
    "LimitStep",
    "DedupeStep",
    # Registry functions
    "STEP_REGISTRY",
    "get_step_class",
    "list_available_steps",
]
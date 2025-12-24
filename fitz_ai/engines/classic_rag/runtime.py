# fitz_ai/engines/classic_rag/runtime.py
"""
Classic RAG Runtime - Canonical entry point for Classic RAG execution.

This module provides the single, stable way to execute Classic RAG queries.
All other entry points (CLI, API, etc.) should route through this runtime.
"""

from typing import Any, Dict, Optional

from fitz_ai.core import Answer, Constraints, Provenance
from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config
from fitz_ai.engines.classic_rag.pipeline.engine import RAGPipeline


def run_classic_rag(
    query: str,
    config: Optional[ClassicRagConfig] = None,
    config_path: Optional[str] = None,
    constraints: Optional[Constraints] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Answer:
    """
    Execute a Classic RAG query.

    This is the canonical entry point for all Classic RAG execution.

    Args:
        query: The question text
        config: Optional pre-loaded ClassicRagConfig
        config_path: Optional path to config file
        constraints: Optional query-time constraints
        metadata: Optional engine-specific metadata

    Returns:
        Answer object with generated text and source provenance
    """
    if config is None:
        config = load_config(config_path)

    pipeline = RAGPipeline.from_config(config)
    rag_answer = pipeline.run(query)

    # Convert to core Answer type
    provenance = []
    if hasattr(rag_answer, "sources") and rag_answer.sources:
        for source_ref in rag_answer.sources:
            prov = Provenance(
                source_id=getattr(source_ref, "source_id", str(source_ref)),
                excerpt=getattr(source_ref, "text", None),
                metadata=getattr(source_ref, "metadata", {}),
            )
            provenance.append(prov)

    answer_metadata = {
        "engine": "classic_rag",
        "query_text": query,
    }

    if hasattr(rag_answer, "metadata") and rag_answer.metadata:
        answer_metadata["rag_metadata"] = rag_answer.metadata

    return Answer(
        text=rag_answer.answer if hasattr(rag_answer, "answer") else str(rag_answer),
        provenance=provenance,
        metadata=answer_metadata,
    )


def create_classic_rag_engine(
    config: Optional[ClassicRagConfig] = None,
    config_path: Optional[str] = None,
) -> RAGPipeline:
    """
    Create and return a Classic RAG pipeline instance.

    Args:
        config: Optional pre-loaded ClassicRagConfig
        config_path: Optional path to config file

    Returns:
        Initialized RAGPipeline instance
    """
    if config is None:
        config = load_config(config_path)

    return RAGPipeline.from_config(config)


# Convenience alias
run = run_classic_rag


# =============================================================================
# AUTO-REGISTRATION WITH GLOBAL REGISTRY
# =============================================================================


def _register_classic_rag_engine():
    """Register Classic RAG engine with the global registry."""
    from fitz_ai.runtime.registry import EngineRegistry

    def classic_rag_factory(config):
        """Factory for creating Classic RAG pipelines."""
        if config is None:
            config = load_config(None)
        elif isinstance(config, dict):
            config = ClassicRagConfig.from_dict(config)

        return RAGPipeline.from_config(config)

    try:
        registry = EngineRegistry.get_global()
        registry.register(
            name="classic_rag",
            factory=classic_rag_factory,
            description="Retrieval-augmented generation using vector search and LLM synthesis",
            config_type=ClassicRagConfig,
        )
    except ValueError:
        pass  # Already registered


# Auto-register on import
_register_classic_rag_engine()

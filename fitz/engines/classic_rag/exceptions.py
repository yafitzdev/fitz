# fitz/engines/classic_rag/exceptions.py
"""
All exceptions for the Classic RAG engine.

Hierarchy:
    EngineError (from fitz.core)
    ├── PipelineError - Pipeline orchestration failures
    │   └── RGSGenerationError - RGS prompt/answer construction failures
    ├── RetrieverError - Retrieval failures
    │   ├── EmbeddingError - Embedding model failures
    │   ├── VectorSearchError - Vector DB failures
    │   └── RerankError - Reranking failures
    ├── LLMError - LLM call failures
    │   └── LLMResponseError - Invalid LLM response
    └── ConfigError - Configuration failures
"""
from fitz.core.exceptions import EngineError, GenerationError

# =============================================================================
# Pipeline Errors
# =============================================================================


class PipelineError(EngineError):
    """Pipeline orchestration failed."""

    pass


class RGSGenerationError(PipelineError, GenerationError):
    """RGS prompt generation or answer synthesis failed."""

    pass


# =============================================================================
# Retrieval Errors
# =============================================================================


class RetrieverError(EngineError):
    """General retrieval failure."""

    pass


class EmbeddingError(RetrieverError):
    """Embedding model failed (API failure, invalid input, etc.)."""

    pass


class VectorSearchError(RetrieverError):
    """Vector database lookup failed."""

    pass


class RerankError(RetrieverError):
    """Reranking failed or returned invalid data."""

    pass


# =============================================================================
# LLM Errors
# =============================================================================


class LLMError(EngineError):
    """LLM call failed."""

    pass


class LLMResponseError(LLMError):
    """LLM returned invalid or unparseable response."""

    pass


# =============================================================================
# Config Errors
# =============================================================================


class ConfigError(EngineError):
    """Configuration error."""

    pass


__all__ = [
    # Pipeline
    "PipelineError",
    "RGSGenerationError",
    # Retrieval
    "RetrieverError",
    "EmbeddingError",
    "VectorSearchError",
    "RerankError",
    # LLM
    "LLMError",
    "LLMResponseError",
    # Config
    "ConfigError",
]

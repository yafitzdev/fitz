from __future__ import annotations


class RetrieverError(Exception):
    """Base class for all retrieval-related errors."""


class VectorSearchError(RetrieverError):
    """Raised when vector search against the backend fails."""


class RerankError(RetrieverError):
    """Raised when reranking retrieved chunks fails."""

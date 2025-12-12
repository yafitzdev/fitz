from .base import FitzRAGError

class RetrieverError(FitzRAGError):
    """General retriever failure."""
    pass


class EmbeddingError(RetrieverError):
    """Embedding model failed (e.g., API failure, invalid input)."""
    pass


class VectorSearchError(RetrieverError):
    """Vector database lookup failure."""
    pass


class RerankError(RetrieverError):
    """Reranking failed or returned invalid data."""
    pass

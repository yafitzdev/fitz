# src/fitz_rag/llm/embedding_client.py
"""
Embedding client wrapper for fitz-rag.

This defines a minimal interface for embedding providers.
You can implement:
- OpenAI embeddings
- Local embedding models
- Azure OpenAI
- HuggingFace embeddings
- Any custom provider

As long as implement embed(text) -> list[float].
"""

from __future__ import annotations

from typing import List, Protocol

# NEW – unified config defaults for real embedding clients
from fitz_rag.config import get_config
_cfg = get_config()
DEFAULT_EMBED_MODEL = _cfg.get("models", {}).get("embedding", "text-embedding-3-large")


class EmbeddingClient(Protocol):
    """
    Any embedding provider must implement this Protocol.
    """

    def embed(self, text: str) -> List[float]:
        ...


class DummyEmbeddingClient:
    """
    A minimal embedding stub for testing.

    WARNING: This is not a real embedding model. It only
    returns a fixed-length dummy vector to verify integration.
    """

    def __init__(self, dim: int = 10):
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        # Creates a pseudo-vector based on hash for deterministic testing.
        base = abs(hash(text)) % 997
        return [(base + i) / 997.0 for i in range(self.dim)]

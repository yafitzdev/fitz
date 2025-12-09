# src/fitz_rag/llm/embedding_client.py
"""
Embedding client implementations for fitz-rag.

This module defines a minimal EmbeddingClient protocol plus:
- CohereEmbeddingClient: real embeddings via Cohere's v2 Embed API
- DummyEmbeddingClient: lightweight stub for tests

The goal is to keep the rest of the stack provider-agnostic. All callers
only depend on `EmbeddingClient.embed(text) -> list[float]`.

Environment variables
---------------------
COHERE_API_KEY          # required for CohereEmbeddingClient (unless api_key passed)
COHERE_EMBED_MODEL      # optional, overrides default model name
COHERE_EMBED_INPUT_TYPE # optional, e.g. "search_query" or "search_document"
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Protocol, Optional
import os

try:  # Cohere is optional at import time
    import cohere
except ImportError:  # pragma: no cover - handled lazily at runtime
    cohere = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# EmbeddingClient protocol
# ---------------------------------------------------------------------------


class EmbeddingClient(Protocol):
    """
    Minimal interface for embedding providers.

    Any concrete implementation must implement `embed(text) -> List[float]`.
    """

    def embed(self, text: str) -> List[float]:
        ...


# ---------------------------------------------------------------------------
# Cohere implementation
# ---------------------------------------------------------------------------


@dataclass
class CohereEmbeddingClient:
    """
    Cohere embedding wrapper.

    By default this uses:
      - model:       embed-english-v3.0  (good quality & pricing)
      - input_type:  "search_query"      (for queries; use "search_document" for corpus)
      - embedding_type: "float"

    You can override the model and input_type via constructor arguments or
    environment variables.
    """

    api_key: Optional[str] = None
    model: Optional[str] = None
    input_type: Optional[str] = None
    output_dimension: Optional[int] = None  # None = model default

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError(
                "cohere is not installed. Run `pip install cohere` inside your environment."
            )

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it in your environment or pass api_key=... to CohereEmbeddingClient."
            )

        # Resolve defaults from env if not explicitly passed
        self.model = (
            self.model
            or os.getenv("COHERE_EMBED_MODEL")
            or "embed-english-v3.0"
        )
        self.input_type = (
            self.input_type
            or os.getenv("COHERE_EMBED_INPUT_TYPE")
            or "search_query"
        )

        # v2 client
        self._client = cohere.ClientV2(api_key=key)

    def embed(self, text: str) -> List[float]:
        """
        Create a single embedding for the given text.

        Returns a list[float] suitable for Qdrant or any other vector store.
        """
        # Cohere v2 Embed API returns an object with embeddings.float
        kwargs = {
            "texts": [text],
            "model": self.model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }
        if self.output_dimension is not None:
            kwargs["output_dimension"] = self.output_dimension

        res = self._client.embed(**kwargs)
        # res.embeddings.float is a list of vectors (one per input text)
        return res.embeddings.float[0]


# ---------------------------------------------------------------------------
# Dummy implementation (for tests / offline work)
# ---------------------------------------------------------------------------


@dataclass
class DummyEmbeddingClient:
    """
    A minimal embedding stub for testing.

    WARNING: This is not a real embedding model. It only
    returns a fixed-length dummy vector to verify integration.
    """

    dim: int = 10

    def embed(self, text: str) -> List[float]:
        # Creates a pseudo-vector based on hash for deterministic testing.
        base = abs(hash(text)) % 997
        return [(base + i) / 997.0 for i in range(self.dim)]

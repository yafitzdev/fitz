# fitz/llm/embedding/plugins/cohere.py
"""
Cohere embedding plugin using centralized HTTP client and credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from fitz.core.http import (
    APIError,
    HTTPClientNotAvailable,
    create_api_client,
    handle_api_error,
    raise_for_status,
)
from fitz.engines.classic_rag.errors.llm import EmbeddingError
from fitz.llm.credentials import CredentialError, resolve_api_key


@dataclass
class CohereEmbeddingClient:
    """
    Cohere embedding plugin using centralized HTTP client and credentials.

    Required:
        - COHERE_API_KEY environment variable OR api_key parameter

    Optional:
        - model: Embedding model (default: embed-english-v3.0)
        - input_type: Type of input (default: search_document for ingestion)
        - output_dimension: Dimension reduction (default: None - full dimensions)
    """

    plugin_name: str = "cohere"
    plugin_type: str = "embedding"

    api_key: str | None = None
    model: str | None = None
    input_type: str | None = None
    output_dimension: int | None = None
    base_url: str = "https://api.cohere.ai/v1"

    # Internal client (not a dataclass field)
    _client: Any = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        # Use centralized credential resolution
        try:
            key = resolve_api_key(
                provider="cohere",
                config={"api_key": self.api_key} if self.api_key else None,
            )
        except CredentialError as e:
            raise RuntimeError(str(e)) from e

        self._api_key = key

        # Set defaults (still allow env var override for model-specific settings)
        self.model = self.model or os.getenv("COHERE_EMBED_MODEL") or "embed-english-v3.0"
        self.input_type = (
            self.input_type or os.getenv("COHERE_EMBED_INPUT_TYPE") or "search_document"
        )

        # Create HTTP client using centralized factory
        try:
            self._client = create_api_client(
                base_url=self.base_url,
                api_key=self._api_key,
                timeout_type="embedding",  # 30s timeout
            )
        except HTTPClientNotAvailable:
            raise RuntimeError(
                "httpx is required for Cohere plugin. " "Install with: pip install httpx"
            )

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            List of floats representing the embedding vector

        Raises:
            EmbeddingError: If the API request fails
        """
        payload: dict[str, object] = {
            "texts": [text],
            "model": self.model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }

        if self.output_dimension is not None:
            payload["truncate"] = "END"

        try:
            response = self._client.post("/embed", json=payload)
            raise_for_status(response, provider="cohere", endpoint="/embed")

            data = response.json()

            # Extract embedding from response
            embeddings_data = data.get("embeddings", {})

            # Handle both v1 and v2 API response formats
            if isinstance(embeddings_data, dict):
                float_embeddings = embeddings_data.get("float", [])
                if float_embeddings and isinstance(float_embeddings, list) and float_embeddings[0]:
                    return float_embeddings[0]
            elif isinstance(embeddings_data, list):
                if embeddings_data and embeddings_data[0]:
                    return embeddings_data[0]

            raise EmbeddingError("No embedding returned from Cohere API")

        except APIError as exc:
            raise EmbeddingError(str(exc)) from exc

        except EmbeddingError:
            raise

        except Exception as exc:
            error = handle_api_error(exc, provider="cohere", endpoint="/embed")
            raise EmbeddingError(str(error)) from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client") and self._client:
            try:
                self._client.close()
            except Exception:
                pass

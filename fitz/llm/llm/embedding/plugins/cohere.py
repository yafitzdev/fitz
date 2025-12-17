# core/llm/embedding/plugins/cohere.py
from __future__ import annotations

import os
from dataclasses import dataclass

from fitz.engines.classic_rag.errors.llm import EmbeddingError

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore


@dataclass
class CohereEmbeddingClient:
    """
    Cohere embedding plugin using direct HTTP requests.

    Zero dependencies beyond httpx (which is already required by fitz core).

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

    def __post_init__(self) -> None:
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Cohere plugin. " "Install with: pip install httpx"
            )

        # Get API key
        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it as an environment variable or pass api_key parameter."
            )
        self._api_key = key

        # Set defaults
        self.model = self.model or os.getenv("COHERE_EMBED_MODEL") or "embed-english-v3.0"
        self.input_type = (
            self.input_type or os.getenv("COHERE_EMBED_INPUT_TYPE") or "search_document"
        )

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
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
        # Build request payload
        payload: dict[str, object] = {
            "texts": [text],
            "model": self.model,
            "input_type": self.input_type,
            "embedding_types": ["float"],
        }

        if self.output_dimension is not None:
            payload["truncate"] = "END"  # Required when using output_dimension
            # Note: output_dimension is only available in v2 API
            # For v1, we'll just use full dimensions

        try:
            response = self._client.post("/embed", json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract embedding from response
            # Response structure: {"embeddings": {"float": [[0.1, 0.2, ...]]}}
            embeddings_data = data.get("embeddings", {})

            # Handle both v1 and v2 API response formats
            if isinstance(embeddings_data, dict):
                # v1 API format: {"embeddings": {"float": [[...]]}}
                float_embeddings = embeddings_data.get("float", [])
                if float_embeddings and isinstance(float_embeddings, list) and float_embeddings[0]:
                    return float_embeddings[0]
            elif isinstance(embeddings_data, list):
                # v2 API format: {"embeddings": [[...]]}
                if embeddings_data and embeddings_data[0]:
                    return embeddings_data[0]

            raise EmbeddingError(f"No embedding returned from Cohere API")

        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_data = exc.response.json()
                error_detail = f": {error_data.get('message', '')}"
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"

            raise EmbeddingError(
                f"Cohere API request failed with status {exc.response.status_code}{error_detail}"
            ) from exc

        except Exception as exc:
            raise EmbeddingError(f"Failed to embed text: {exc}") from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"

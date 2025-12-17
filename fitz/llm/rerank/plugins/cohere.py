# fitz/llm/rerank/plugins/cohere.py
"""
Cohere rerank plugin using centralized HTTP client.
"""

from __future__ import annotations

import os

from fitz.core.http import (
    create_api_client,
    raise_for_status,
    handle_api_error,
    APIError,
    HTTPClientNotAvailable,
)
from fitz.engines.classic_rag.models.chunk import Chunk


class CohereRerankClient:
    """
    Cohere rerank plugin using centralized HTTP client.

    Required:
        - COHERE_API_KEY environment variable OR api_key parameter

    Optional:
        - model: Rerank model (default: rerank-english-v3.0)
    """

    plugin_name = "cohere"
    plugin_type = "rerank"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
        base_url: str = "https://api.cohere.ai/v1",
    ) -> None:
        # Get API key
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it as an environment variable or pass api_key parameter."
            )

        self.model = model
        self.base_url = base_url

        # Create HTTP client using centralized factory
        try:
            self._client = create_api_client(
                base_url=self.base_url,
                api_key=key,
                timeout_type="rerank",  # 30s timeout
            )
        except HTTPClientNotAvailable:
            raise RuntimeError(
                "httpx is required for Cohere plugin. "
                "Install with: pip install httpx"
            )

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """
        Rerank chunks based on relevance to query.

        Args:
            query: The search query
            chunks: List of chunks to rerank

        Returns:
            Reranked list of chunks (sorted by relevance, highest first)

        Raises:
            RuntimeError: If the API request fails
        """
        if not chunks:
            return []

        documents = [chunk.content for chunk in chunks]

        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }

        try:
            response = self._client.post("/rerank", json=payload)
            raise_for_status(response, provider="cohere", endpoint="/rerank")

            data = response.json()

            results = data.get("results", [])

            sorted_results = sorted(
                results,
                key=lambda r: r.get("relevance_score", 0.0),
                reverse=True,
            )

            return [chunks[r["index"]] for r in sorted_results]

        except APIError as exc:
            raise RuntimeError(str(exc)) from exc

        except Exception as exc:
            error = handle_api_error(exc, provider="cohere", endpoint="/rerank")
            raise RuntimeError(str(error)) from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass
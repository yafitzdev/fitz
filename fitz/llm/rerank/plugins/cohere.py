# core/llm/rerank/plugins/cohere.py
from __future__ import annotations

import os

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore

from fitz.engines.classic_rag.models.chunk import Chunk


class CohereRerankClient:
    """
    Cohere rerank plugin using direct HTTP requests.

    Zero dependencies beyond httpx (which is already required by fitz core).

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
        if httpx is None:
            raise RuntimeError(
                "httpx is required for Cohere plugin. " "Install with: pip install httpx"
            )

        # Get API key
        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. "
                "Set it as an environment variable or pass api_key parameter."
            )
        self._api_key = key

        self.model = model
        self.base_url = base_url

        # Create HTTP client
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
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

        # Extract text from chunks
        documents = [chunk.content for chunk in chunks]

        # Build request payload
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
        }

        try:
            response = self._client.post("/rerank", json=payload)
            response.raise_for_status()

            data = response.json()

            # Extract results
            # Response structure: {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
            results = data.get("results", [])

            # Sort by relevance score (highest first)
            sorted_results = sorted(
                results,
                key=lambda r: r.get("relevance_score", 0.0),
                reverse=True,
            )

            # Return chunks in reranked order
            return [chunks[r["index"]] for r in sorted_results]

        except httpx.HTTPStatusError as exc:
            error_detail = ""
            try:
                error_data = exc.response.json()
                error_detail = f": {error_data.get('message', '')}"
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"

            raise RuntimeError(
                f"Cohere API request failed with status {exc.response.status_code}{error_detail}"
            ) from exc

        except Exception as exc:
            raise RuntimeError(f"Failed to rerank: {exc}") from exc

    def __del__(self):
        """Clean up HTTP client on deletion."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception as e:
                # Failed to parse error response
                error_detail = f" (response parse failed: {e})"

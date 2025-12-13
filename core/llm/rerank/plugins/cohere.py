# core/llm/rerank/plugins/cohere.py
from __future__ import annotations

import os

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore

from core.models.chunk import Chunk


class CohereRerankClient:
    plugin_name = "cohere"
    plugin_type = "rerank"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        self.model = model
        self._client = cohere.ClientV2(api_key=key)

    def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        docs = [c.content for c in chunks]
        response = self._client.rerank(model=self.model, query=query, documents=docs)

        results = sorted(
            response.results,
            key=lambda r: r.relevance_score,
            reverse=True,
        )
        return [chunks[r.index] for r in results]

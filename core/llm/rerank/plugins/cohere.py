# core/llm/plugins/cohere.py
from __future__ import annotations

from typing import List
import os

from rag.exceptions.retriever import RerankError
from rag.models.chunk import Chunk
from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereRerankClient:
    plugin_name = "cohere"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.model = model
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        try:
            self._client = cohere.ClientV2(api_key=self.api_key)
        except Exception as exc:
            raise RerankError("Failed to initialize Cohere rerank client") from exc

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        try:
            docs = [c.content for c in chunks]

            response = self._client.rerank(
                model=self.model,
                query=query,
                documents=docs,
            )

            sorted_results = sorted(
                response.results,
                key=lambda r: r.relevance_score,
                reverse=True,
            )

            return [chunks[r.index] for r in sorted_results]

        except Exception as exc:
            raise RerankError("Reranking failed") from exc


register_llm_plugin(
    CohereRerankClient,
    plugin_name="cohere",
    plugin_type="rerank",
)

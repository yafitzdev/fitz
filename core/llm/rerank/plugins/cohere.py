# ============================
# File: src/fitz_stack/llm/rerank/plugins/cohere.py
# ============================
"""
Cohere reranking client for Fitz-RAG (ClientV2 version).
"""

from __future__ import annotations

from typing import Any, List
import os

from rag.exceptions.retriever import RerankError
from core.llm.registry import register_llm_plugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


class CohereRerankClient:
    plugin_name = "cohere"

    def __init__(
        self,
        plugin_name: str = "cohere",
        api_key: str | None = None,
        model: str = "rerank-english-v3.0",
    ) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        self.plugin_name = plugin_name
        self.model = model

        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError("COHERE_API_KEY is not set for CohereRerankClient")

        try:
            self._client = cohere.ClientV2(api_key=self.api_key)
        except Exception as e:
            raise RerankError("Failed to initialize Cohere rerank client") from e

    def rerank(self, query: str, chunks: List[Any]) -> List[int]:
        try:
            docs: List[str] = []

            for idx, c in enumerate(chunks):
                if isinstance(c, str):
                    docs.append(c)
                elif isinstance(c, dict) and isinstance(c.get("text"), str):
                    docs.append(c["text"])
                else:
                    text = getattr(c, "text", None)
                    if not isinstance(text, str):
                        raise TypeError(f"Unsupported chunk format at index {idx}: {type(c)}")
                    docs.append(text)

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
            return [r.index for r in sorted_results]

        except Exception as e:
            raise RerankError("Reranking failed") from e


# Register plugin on import
register_llm_plugin(
    CohereRerankClient,
    plugin_name="cohere",
    plugin_type="rerank",
)

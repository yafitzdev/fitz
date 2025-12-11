from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os

from fitz_rag.core import Chunk
from fitz_rag.exceptions.retriever import RerankError
from fitz_rag.llm.rerank.base import RerankPlugin

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereRerankClient(RerankPlugin):
    """
    Cohere reranking plugin for fitz-rag.

    Responsibilities:
    - Call Cohere API to rerank documents
    - Convert API result into ordered Chunk list

    Exposes plugin_name = "cohere" for auto-discovery.
    """

    plugin_name: str = "cohere"

    api_key: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("Install cohere: `pip install cohere`")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set")

        self.model = (
            self.model
            or os.getenv("COHERE_RERANK_MODEL")
            or "rerank-english-v3.0"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise RerankError("Failed to initialize Cohere rerank client") from e

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        try:
            docs = [c.text for c in chunks]
            response = self._client.rerank(
                query=query,
                documents=docs,
                model=self.model,
            )

            # Sort in descending score order
            ordered = sorted(
                zip(chunks, response.results),
                key=lambda x: x[1].relevance_score,
                reverse=True,
            )

            return [pair[0] for pair in ordered]

        except Exception as e:
            raise RerankError("Reranking failed") from e

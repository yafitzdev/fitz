from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
import os

from fitz_rag.exceptions.retriever import RerankError

try:
    import cohere
except ImportError:
    cohere = None  # type: ignore


@dataclass
class CohereRerankClient:
    """
    Cohere Rerank API plugin for fitz-rag.

    Responsibilities:
    - Accept plain text documents
    - Return a list of ranking indices
    - Leave chunk handling to RerankEngine
    """

    api_key: Optional[str] = None
    model: Optional[str] = None

    def __post_init__(self) -> None:
        if cohere is None:
            raise RuntimeError("cohere is not installed. Run `pip install cohere`.")

        key = self.api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("COHERE_API_KEY is not set.")

        self.model = (
            self.model
            or os.getenv("COHERE_RERANK_MODEL")
            or "rerank-v3.5"
        )

        try:
            self._client = cohere.ClientV2(api_key=key)
        except Exception as e:
            raise RerankError("Failed to initialize Cohere Rerank client") from e

    # ---------------------------------------------------------
    # Plugin API
    # ---------------------------------------------------------
    def rerank(
        self,
        query: str,
        documents: List[str],
        top_n: Optional[int] = None
    ) -> List[int]:

        if not documents:
            return []

        n = top_n if top_n is not None else len(documents)

        try:
            res = self._client.rerank(
                model=self.model,
                query=query,
                documents=documents,
                top_n=n,
            )
        except Exception as e:
            raise RerankError("Rerank request failed") from e

        # Extract numerical ordering
        try:
            return [r.index for r in res.results]
        except Exception as e:
            raise RerankError("Malformed rerank response returned by provider") from e

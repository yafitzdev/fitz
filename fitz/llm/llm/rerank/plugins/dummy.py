from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class DummyRerankClient:
    """
    Deterministic test reranker.

    Sorts documents by descending text length.
    This plugin is intentionally simple and stable for testing RerankEngine.
    """

    def rerank(self, query: str, documents: List[str], top_n: Optional[int] = None) -> List[int]:

        if not documents:
            return []

        # Longest documents first (deterministic)
        indices = sorted(range(len(documents)), key=lambda i: len(documents[i]), reverse=True)

        if top_n is not None:
            indices = indices[:top_n]

        return indices

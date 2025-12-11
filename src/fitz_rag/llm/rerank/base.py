from __future__ import annotations
from typing import Protocol, List

from fitz_rag.core import Chunk


class RerankPlugin(Protocol):
    """
    Protocol for reranking plugins.

    Any reranker (Cohere, custom scoring, metadata-based ranking, etc.)
    should implement this interface.

    Plugins typically live in:
        fitz_rag.llm.rerank.plugins.<name>

    And declare a unique:
        plugin_name: str
    """

    # Required for auto-registration
    # plugin_name: str = "unique-name"

    def rerank(self, query: str, chunks: List[Chunk]) -> List[Chunk]:
        """
        Rerank a list of chunks given the input query.
        Must return a NEW list or a mutated copy.
        """
        ...

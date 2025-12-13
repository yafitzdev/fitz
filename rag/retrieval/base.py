# rag/retrieval/base.py

from __future__ import annotations

from typing import List, Protocol, runtime_checkable

from core.models.chunk import Chunk


@runtime_checkable
class RetrievalPlugin(Protocol):
    """
    Protocol for retrieval plugins/strategies.

    Any retrieval strategy (dense, BM25, hybrid, multi-query, etc.)
    must implement this interface.

    Plugins typically live in:
        rag.retrieval.plugins.<name>
    and declare a unique `plugin_name` attribute.
    """

    def retrieve(self, query: str) -> List[Chunk]:
        """
        Run retrieval for the given user query and return a list of chunks.
        """
        ...

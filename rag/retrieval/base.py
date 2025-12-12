# rag/retrieval/base.py

from __future__ import annotations

from typing import Protocol, List, runtime_checkable

from rag.models.chunk import Chunk


@runtime_checkable
class RetrievalPlugin(Protocol):
    """
    Protocol for retrieval plugins/strategies.

    Any retrieval strategy (dense, BM25, hybrid, multi-query, etc.)
    must implement this interface.

    Plugins typically live in:
        fitz_rag.retrieval.plugins.<name>
    and declare a unique `plugin_name` attribute.
    """

    # Each plugin should define a class attribute:
    #   plugin_name: str = "<unique-name>"
    #
    # This is used by the auto-discovery registry.

    def retrieve(self, query: str) -> List[Chunk]:
        """
        Run retrieval for the given user query and return a list of chunks.
        """
        ...

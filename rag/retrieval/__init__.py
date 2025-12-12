# rag/retrieval/__init__.py
"""
Retrieval subsystem public API.

Keep this module import-light to avoid circular imports.
We provide lazy re-exports for convenience.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from rag.retrieval.base import RetrievalPlugin as RetrievalPlugin
    from rag.retrieval.engine import RetrieverEngine as RetrieverEngine

__all__ = ["RetrievalPlugin", "RetrieverEngine"]


def __getattr__(name: str) -> Any:
    if name == "RetrievalPlugin":
        from rag.retrieval.base import RetrievalPlugin as _RetrievalPlugin

        return _RetrievalPlugin

    if name == "RetrieverEngine":
        from rag.retrieval.engine import RetrieverEngine as _RetrieverEngine

        return _RetrieverEngine

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

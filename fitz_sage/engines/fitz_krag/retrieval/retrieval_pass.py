# fitz_sage/engines/fitz_krag/retrieval/retrieval_pass.py
"""
One retrieval pass — candidate generation, fusion, precision rerank, read.

`RetrievalPass` is the unit the tiered retrieval stack is built from:

    Tier 1  candidate generation  ┐
    Tier 2  cross-strategy fusion ┘── RetrievalRouter.retrieve()
    Tier 3  precision rerank       ── AddressReranker.rerank()
    Tier 4  read content           ── ContentReader.read()

Query in, `ReadResult`s out. A single-hop query runs one pass; the
multi-hop controller loops it. Reranking lives *inside* the pass, so it
runs on every query regardless of how many hops there are.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import ReadResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
    from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker
    from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter


class RetrievalPass:
    """Tiers 1-4 of the retrieval stack as one composable unit."""

    def __init__(
        self,
        router: "RetrievalRouter",
        reranker: "AddressReranker | None",
        reader: "ContentReader",
        config: "FitzKragConfig",
    ) -> None:
        self._router = router
        self._reranker = reranker
        self._reader = reader
        self._config = config

    def run(
        self,
        query: str,
        profile: Any = None,
        *,
        exclude: set[tuple[str, str]] | None = None,
        rewrite_result: Any = None,
        progress: "Callable[[str], None] | None" = None,
    ) -> list[ReadResult]:
        """Run one retrieval pass: retrieve -> drop excluded -> rerank -> read.

        Args:
            query: the retrieval query (rewritten or bridge query, not raw).
            profile: the RetrievalProfile carrying gates + strategy weights.
            exclude: address keys ``(source_id, location)`` to drop before
                reranking — used by multi-hop to skip already-read addresses.
            rewrite_result: the QueryRewriter result, forwarded to the router
                so it can reuse decomposed query variations.
            progress: optional status callback, forwarded to the router.

        Returns:
            Read results for the surviving addresses (``<= rerank_k`` when a
            reranker is configured).
        """
        addresses = self._router.retrieve(
            query, profile, rewrite_result=rewrite_result, progress=progress
        )
        if exclude:
            addresses = [a for a in addresses if (a.source_id, a.location) not in exclude]
        if not addresses:
            return []
        if self._reranker is not None:
            addresses = self._reranker.rerank(query, addresses)
        return self._reader.read(addresses, self._config.top_read)


__all__ = ["RetrievalPass"]

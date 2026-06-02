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

import re
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import ReadResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
    from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker
    from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter

_BROAD_OVERVIEW_TERMS = (
    "summary",
    "overview",
    "roadmap",
    "report",
    "quarterly",
    "annual",
    "executive",
    "key metrics",
    "feedback",
)
_CONTROL_SURFACE_TERMS = {"test", "tests", "case", "cases", "fixture", "fixtures", "readme"}
_CONTROL_SURFACE_MARKERS = (
    "keyword_test",
    "test_cases",
    "/test",
    "\\test",
    "_test",
    "fixture",
    "fixtures",
    "readme",
    "source_dir",
    "collections/",
    ".fitz",
)


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
        addresses = _apply_broad_corpus_prior(query, addresses, profile)
        addresses = _enforce_broad_file_diversity(addresses, profile)
        return self._reader.read(addresses, self._config.top_read)


def _apply_broad_corpus_prior(query: str, addresses: list[Any], profile: Any = None) -> list[Any]:
    """For corpus overviews, prefer overview files over test/control surfaces."""
    if profile is None:
        return addresses
    if (
        getattr(profile, "specificity", "") != "broad"
        and getattr(profile, "answer_type", "") != "exploratory"
    ):
        return addresses

    query_terms = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
    if query_terms & _CONTROL_SURFACE_TERMS:
        return addresses

    scored: list[tuple[int, int, Any]] = []
    for index, address in enumerate(addresses):
        scored.append((_broad_corpus_priority(address), index, address))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [address for _, _, address in scored]


def _broad_corpus_priority(address: Any) -> int:
    """Score address-level corpus overview usefulness before final reading."""
    haystack = _address_text(address)
    priority = 0
    if any(term in haystack for term in _BROAD_OVERVIEW_TERMS):
        priority += 1
    if any(marker in haystack for marker in _CONTROL_SURFACE_MARKERS):
        priority -= 2
    return priority


def _address_text(address: Any) -> str:
    """Combine stable address fields used by broad-corpus ranking priors."""
    metadata = getattr(address, "metadata", {}) or {}
    parts = [
        getattr(address, "source_id", ""),
        getattr(address, "location", ""),
        getattr(address, "summary", ""),
        str(metadata.get("source_path", "")),
        str(metadata.get("disk_path", "")),
    ]
    return " ".join(part for part in parts if part).lower().replace("\\", "/")


def _enforce_broad_file_diversity(addresses: list[Any], profile: Any = None) -> list[Any]:
    """For exploratory queries, defer repeated hits from the same file."""
    if profile is None:
        return addresses
    if (
        getattr(profile, "specificity", "") != "broad"
        and getattr(profile, "answer_type", "") != "exploratory"
    ):
        return addresses

    seen: set[str] = set()
    promoted: list[Any] = []
    deferred: list[Any] = []
    for address in addresses:
        if address.source_id in seen:
            deferred.append(address)
            continue
        seen.add(address.source_id)
        promoted.append(address)
    return promoted + deferred


__all__ = ["RetrievalPass"]

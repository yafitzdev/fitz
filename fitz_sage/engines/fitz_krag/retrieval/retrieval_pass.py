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
import time
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.evidence_compiler import order_addresses_for_contract
from fitz_sage.engines.fitz_krag.retrieval.trace import addresses_trace, read_results_trace
from fitz_sage.engines.fitz_krag.types import ReadResult

if TYPE_CHECKING:
    from collections.abc import Callable

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.retrieval.reader import ContentReader
    from fitz_sage.engines.fitz_krag.retrieval.reranker import AddressReranker
    from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter

_BROAD_PATH_WEIGHTS = {
    "summary": 3,
    "roadmap": 3,
    "quarterly": 2,
    "report": 1,
    "annual": 2,
}
_BROAD_LOCATION_WEIGHTS = {
    "executive summary": 2,
    "key metrics": 2,
    "overview": 1,
}
_BROAD_SHARED_WEIGHTS = {
    "feedback": 1,
    "executive": 2,
}
_CONTROL_SURFACE_TERMS = {
    "test",
    "tests",
    "case",
    "cases",
    "fixture",
    "fixtures",
    "readme",
    "query",
    "queries",
    "prompt",
    "prompts",
}
_CONTROL_SURFACE_MARKERS = (
    "keyword_test",
    "test_cases",
    "near_duplicate",
    "poisoning",
    "/artifacts/",
    "/artifcats/",
    "/queries",
    "queries.",
    "_queries",
    "/test",
    "\\test",
    "_test",
    "fixture",
    "fixtures",
    "formal_eval_harness",
    "readme",
    "source_dir",
    "collections/",
    ".fitz",
)
_BROAD_GROUP_TARGET = 3
_BROAD_GROUP_LOOKAHEAD = 16


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
        self.last_timings: dict[str, float] = {}
        self.last_trace: dict[str, Any] = {}

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
        self.last_timings = {}
        self.last_trace = {"query": query}

        t0 = time.perf_counter()
        addresses = self._router.retrieve(
            query, profile, rewrite_result=rewrite_result, progress=progress
        )
        self.last_timings["recall"] = time.perf_counter() - t0
        recall_addresses = list(addresses)
        router_trace = dict(getattr(self._router, "last_trace", {}) or {})
        if exclude:
            addresses = [a for a in addresses if (a.source_id, a.location) not in exclude]
        after_exclude = list(addresses)
        if not addresses:
            self.last_timings["rerank"] = 0.0
            self.last_timings["read"] = 0.0
            self.last_trace = {
                "query": query,
                "profile": _profile_trace(profile),
                "exclude_count": len(exclude or set()),
                "router": router_trace,
                "recall_count": len(recall_addresses),
                "recall": addresses_trace(recall_addresses),
                "after_exclude_count": len(after_exclude),
                "after_exclude": addresses_trace(after_exclude),
                "reranker": {"used": False, "reason": "no_addresses"},
                "final_addresses": [],
                "read_results": [],
                "timings": dict(self.last_timings),
            }
            return []
        if self._reranker is not None:
            candidates = addresses
            t0 = time.perf_counter()
            addresses = self._reranker.rerank(query, addresses)
            self.last_timings["rerank"] = time.perf_counter() - t0
            reranker_trace = dict(getattr(self._reranker, "last_trace", {}) or {})
            addresses = order_addresses_for_contract(
                query,
                candidates,
                addresses,
                profile,
                limit=len(addresses),
            )
            addresses = _ensure_broad_corpus_coverage(query, candidates, addresses, profile)
            addresses = _ensure_concrete_row_coverage(candidates, addresses, profile)
        else:
            self.last_timings["rerank"] = 0.0
            reranker_trace = {"used": False, "reason": "no_reranker"}
        addresses = _apply_broad_corpus_prior(query, addresses, profile)
        addresses = _enforce_broad_file_diversity(addresses, profile)
        addresses = _enforce_broad_group_diversity(query, addresses, profile)
        addresses = _assign_broad_effective_scores(query, addresses, profile)
        final_addresses = list(addresses)
        t0 = time.perf_counter()
        read_limit = getattr(profile, "top_read", self._config.top_read)
        results = self._reader.read(addresses, read_limit)
        self.last_timings["read"] = time.perf_counter() - t0
        self.last_trace = {
            "query": query,
            "profile": _profile_trace(profile),
            "exclude_count": len(exclude or set()),
            "router": router_trace,
            "recall_count": len(recall_addresses),
            "recall": addresses_trace(recall_addresses),
            "after_exclude_count": len(after_exclude),
            "after_exclude": addresses_trace(after_exclude),
            "reranker": reranker_trace,
            "final_address_count": len(final_addresses),
            "final_addresses": addresses_trace(final_addresses),
            "read_limit": read_limit,
            "read_result_count": len(results),
            "read_results": read_results_trace(results),
            "timings": dict(self.last_timings),
        }
        return results


def _profile_trace(profile: Any) -> dict[str, Any]:
    """Serialize the retrieval profile fields most useful for benchmark analysis."""
    if profile is None:
        return {}
    fields = (
        "specificity",
        "answer_type",
        "analysis_type",
        "analysis_confidence",
        "domain",
        "top_k",
        "top_read",
        "strategy_weights",
        "retrieval_modality",
        "retrieval_obligation",
        "required_modalities",
        "keywords",
        "entities",
        "query_variations",
        "comparison_queries",
        "comparison_entities",
        "temporal_references",
        "run_agentic",
        "inject_corpus_summaries",
        "entity_expansion_limit",
    )
    trace: dict[str, Any] = {}
    for field_name in fields:
        if hasattr(profile, field_name):
            trace[field_name] = getattr(profile, field_name)
    return trace


def _ensure_concrete_row_coverage(
    candidates: list[Any],
    ranked: list[Any],
    profile: Any = None,
) -> list[Any]:
    """Keep a strongly matching concrete table row through top-k reranking."""
    if not candidates or not ranked:
        return ranked

    strong_candidates = [
        candidate for candidate in candidates if _concrete_row_strength(candidate) is not None
    ]
    if not strong_candidates:
        return ranked
    candidate = max(
        strong_candidates,
        key=lambda item: (
            _concrete_row_strength(item) or (0.0, 0),
            float(getattr(item, "score", 0.0) or 0.0),
        ),
    )
    candidate_key = _address_identity(candidate)
    if any(_address_identity(address) == candidate_key for address in ranked):
        return ranked

    selected = list(ranked)
    selected_counts: dict[Any, int] = {}
    for address in selected:
        kind = getattr(address, "kind", None)
        selected_counts[kind] = selected_counts.get(kind, 0) + 1
    required_kinds = set(tuple(getattr(profile, "required_modalities", ()) or ()))
    replace_at = next(
        (
            index
            for index in range(len(selected) - 1, -1, -1)
            if selected_counts.get(getattr(selected[index], "kind", None), 0)
            > (
                1
                if getattr(
                    getattr(selected[index], "kind", None),
                    "value",
                    str(getattr(selected[index], "kind", None)),
                )
                in required_kinds
                else 0
            )
        ),
        None,
    )
    if replace_at is None:
        selected.append(candidate)
    else:
        selected[replace_at] = candidate
    return selected


def _concrete_row_strength(address: Any) -> tuple[float, int] | None:
    """Return row-match confidence only for concrete, query-aligned rows."""
    metadata = getattr(address, "metadata", {}) or {}
    row_match = metadata.get("row_match")
    if isinstance(row_match, dict):
        try:
            matched_rows = int(row_match.get("matched_rows", 0) or 0)
        except (TypeError, ValueError):
            matched_rows = 0
        if matched_rows > 0:
            return (1.0, matched_rows)

    row_search = metadata.get("row_search")
    if not isinstance(row_search, dict):
        return None
    try:
        coverage = float(row_search.get("term_coverage", 0.0) or 0.0)
    except (TypeError, ValueError):
        return None
    matched_terms = row_search.get("matched_terms")
    matched_count = len(matched_terms) if isinstance(matched_terms, list) else 0
    if coverage < 0.5 or matched_count < 2:
        return None
    return (coverage, matched_count)


def _address_identity(address: Any) -> tuple[Any, Any, Any]:
    return (
        getattr(address, "kind", None),
        getattr(address, "source_id", None),
        getattr(address, "location", None),
    )


def _apply_broad_corpus_prior(query: str, addresses: list[Any], profile: Any = None) -> list[Any]:
    """For corpus overviews, prefer overview files over test/control surfaces."""
    if not _should_apply_broad_corpus_prior(query, profile):
        return addresses

    scored: list[tuple[int, int, Any]] = []
    for index, address in enumerate(addresses):
        scored.append((_broad_corpus_priority(address), index, address))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [address for _, _, address in scored]


def _ensure_broad_corpus_coverage(
    query: str,
    candidates: list[Any],
    selected: list[Any],
    profile: Any = None,
) -> list[Any]:
    """Preserve high-value overview candidates that reranking may have dropped."""
    if not _should_apply_broad_corpus_prior(query, profile):
        return selected

    selected_keys = {(address.source_id, address.location) for address in selected}
    selected_groups = {
        _broad_group_key(address) for address in selected if not _is_control_surface(address)
    }
    rescued: list[Any] = []
    for address in candidates:
        key = (address.source_id, address.location)
        if key in selected_keys:
            continue
        group = _broad_group_key(address)
        priority_rescue = _broad_corpus_priority(address) > 0
        group_rescue = (
            len(selected_groups) < _BROAD_GROUP_TARGET
            and group not in selected_groups
            and not _is_control_surface(address)
            and _has_broad_overview_signal(address)
        )
        if not priority_rescue and not group_rescue:
            continue
        selected_keys.add(key)
        selected_groups.add(group)
        rescued.append(address)
    return selected + rescued


def _should_apply_broad_corpus_prior(query: str, profile: Any = None) -> bool:
    """Return whether broad corpus ordering should apply to this query."""
    if profile is None:
        return False
    if (
        getattr(profile, "specificity", "") != "broad"
        and getattr(profile, "answer_type", "") != "exploratory"
    ):
        return False

    query_terms = set(re.findall(r"[A-Za-z0-9_]+", query.lower()))
    if query_terms & _CONTROL_SURFACE_TERMS:
        return False
    return True


def _broad_corpus_priority(address: Any) -> int:
    """Score address-level corpus overview usefulness before final reading."""
    path_text = _address_path_text(address)
    location_text = str(getattr(address, "location", "")).lower().replace("\\", "/")
    haystack = f"{path_text} {location_text}"
    priority = 0
    for term, weight in _BROAD_PATH_WEIGHTS.items():
        if term in path_text:
            priority += weight
    for term, weight in _BROAD_LOCATION_WEIGHTS.items():
        if term in location_text:
            priority += weight
    for term, weight in _BROAD_SHARED_WEIGHTS.items():
        if term in haystack:
            priority += weight
    if any(marker in haystack for marker in _CONTROL_SURFACE_MARKERS):
        priority -= 4
    return priority


def _address_path_text(address: Any) -> str:
    """Combine stable path-like address fields used by broad-corpus priors."""
    metadata = getattr(address, "metadata", {}) or {}
    parts = [
        getattr(address, "source_id", ""),
        str(metadata.get("source_path", "")),
        str(metadata.get("disk_path", "")),
    ]
    return " ".join(part for part in parts if part).lower().replace("\\", "/")


def _assign_broad_effective_scores(
    query: str,
    addresses: list[Any],
    profile: Any = None,
) -> list[Any]:
    """Make broad-corpus displayed scores follow the final effective rank."""
    if not _should_apply_broad_corpus_prior(query, profile):
        return addresses

    total = len(addresses)
    rescored: list[Any] = []
    for index, address in enumerate(addresses):
        metadata = dict(address.metadata)
        metadata.setdefault("retrieval_score", address.score)
        metadata["broad_corpus_priority"] = _broad_corpus_priority(address)
        metadata["ranking_score_kind"] = "broad_corpus"
        rescored.append(
            replace(
                address,
                score=(total - index) / total,
                metadata=metadata,
            )
        )
    return rescored


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


def _enforce_broad_group_diversity(
    query: str,
    addresses: list[Any],
    profile: Any = None,
) -> list[Any]:
    """For corpus overview queries, seed the cutoff window with corpus-family coverage."""
    if not _should_apply_broad_corpus_prior(query, profile):
        return addresses
    if len(addresses) <= _BROAD_GROUP_TARGET:
        return addresses

    lookahead = addresses[: min(len(addresses), _BROAD_GROUP_LOOKAHEAD)]
    eligible = [
        address
        for address in lookahead
        if not _is_control_surface(address) and _has_broad_overview_signal(address)
    ]
    available_groups = {_broad_group_key(address) for address in eligible}
    target = min(_BROAD_GROUP_TARGET, len(available_groups))
    if target < 2:
        return addresses

    selected_keys: set[tuple[str, str]] = set()
    selected_groups: set[str] = set()
    promoted: list[Any] = []
    for address in eligible:
        group = _broad_group_key(address)
        if group in selected_groups:
            continue
        promoted.append(address)
        selected_groups.add(group)
        selected_keys.add((address.source_id, address.location))
        if len(promoted) >= target:
            break

    if len(promoted) < 2:
        return addresses

    remainder = [
        address
        for address in addresses
        if (address.source_id, address.location) not in selected_keys
    ]
    return promoted + remainder


def _broad_group_key(address: Any) -> str:
    """Return the top-level corpus family for broad overview diversity."""
    path_text = _primary_address_path(address).replace("\\", "/").strip("/")
    if not path_text:
        return str(getattr(address, "source_id", ""))
    return path_text.split("/", 1)[0].lower()


def _is_control_surface(address: Any) -> bool:
    """Return whether an address is test/control material for broad overview ranking."""
    path_text = _address_path_text(address)
    location_text = str(getattr(address, "location", "")).lower().replace("\\", "/")
    haystack = f"{path_text} {location_text}"
    return any(marker in haystack for marker in _CONTROL_SURFACE_MARKERS)


def _has_broad_overview_signal(address: Any) -> bool:
    """Return whether a candidate is suitable for broad-overview promotion."""
    return _broad_corpus_priority(address) > 0


def _primary_address_path(address: Any) -> str:
    """Return the most stable source path for one address."""
    metadata = getattr(address, "metadata", {}) or {}
    return (
        str(metadata.get("source_path", ""))
        or str(metadata.get("disk_path", ""))
        or str(getattr(address, "source_id", ""))
    )


__all__ = ["RetrievalPass"]

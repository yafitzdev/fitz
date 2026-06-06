# fitz_sage/engines/fitz_krag/query_pipeline.py
"""Query-side retrieval pipeline for KRAG."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from fitz_sage.engines.fitz_krag.query_planner import (
    DeterministicQueryPlanner,
    QueryPlan,
    plan_from_batch_result,
)
from fitz_sage.engines.fitz_krag.retrieval_profile import (
    apply_retrieval_modality_weights,
    build_retrieval_profile,
    query_profile_metadata,
)
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.core import Query
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.types import Address, ReadResult

logger = get_logger(__name__)

_MAX_QUERY_LENGTH = 8000


@dataclass
class RetrievalOutcome:
    """Carrier for the retrieval half of the KRAG pipeline."""

    sanitized: str
    expanded: list["ReadResult"]
    addresses: list["Address"]
    timings: list[tuple[str, float]]
    profile: Any | None = None
    retrieval_query: str = ""
    rewrite_result: Any = None
    query_profile_metadata: dict[str, Any] = field(default_factory=dict)


class QueryPipeline:
    """Analyze, retrieve, read, expand, and table-process one query."""

    def __init__(
        self,
        *,
        config: "FitzKragConfig",
        query_planner: Any,
        query_batcher: Any,
        query_signal_classifier: Any,
        semantic_keyword_batcher: Any,
        retrieval_pass: Any,
        hop_controller: Any,
        expander: Any,
        table_handler: Any,
        retrieval_strategy_scope: Callable[[bool], AbstractContextManager[Any]],
        fast_analyze: Callable[[str], Any],
        needs_detection: Callable[[str], bool],
        build_detection_summary: Callable[[dict[str, Any]], Any],
    ) -> None:
        self._config = config
        self._query_planner = query_planner
        self._query_batcher = query_batcher
        self._query_signal_classifier = query_signal_classifier
        self._semantic_keyword_batcher = semantic_keyword_batcher
        self._retrieval_pass = retrieval_pass
        self._hop_controller = hop_controller
        self._expander = expander
        self._table_handler = table_handler
        self._retrieval_strategy_scope = retrieval_strategy_scope
        self._fast_analyze = fast_analyze
        self._needs_detection = needs_detection
        self._build_detection_summary = build_detection_summary

    def retrieve(
        self,
        query: "Query",
        *,
        progress: Callable[[str], None] | None = None,
        use_query_intelligence: bool | None = None,
        allow_llm_strategies: bool = True,
        execute_table_queries: bool = True,
        expand_context: bool = True,
    ) -> RetrievalOutcome:
        """Run the retrieval half of the KRAG pipeline."""
        sanitized = _sanitize_query(query.text)
        timings: list[tuple[str, float]] = []

        _progress = progress or (lambda _: None)
        _progress("Analyzing query...")

        t0 = time.perf_counter()
        query_signals = self._classify_query_signals(sanitized)
        timings.append(("Pyrrho query signals", time.perf_counter() - t0))

        t0 = time.perf_counter()
        plan = self._prepare_query_plan(
            sanitized,
            query.metadata,
            use_query_intelligence=use_query_intelligence,
        )
        timings.append(("Query prep", time.perf_counter() - t0))

        t0 = time.perf_counter()
        plan = self._add_semantic_query_keywords(sanitized, plan)
        timings.append(("Qwen query keywords", time.perf_counter() - t0))

        profile = build_retrieval_profile(
            plan.analysis,
            plan.detection,
            self._config,
            extended_signals=plan.extended_signals,
            keywords=plan.keywords,
            query_signals=query_signals,
        )
        query_profile = query_profile_metadata(query_signals, profile)

        _progress("Retrieving relevant sources...")
        t0 = time.perf_counter()
        use_multi_hop = (
            allow_llm_strategies and self._hop_controller and self._config.enable_multi_hop
        )
        with self._retrieval_strategy_scope(allow_llm_strategies):
            if use_multi_hop:
                read_results = self._hop_controller.execute(plan.retrieval_query, profile)
            else:
                read_results = self._retrieval_pass.run(
                    plan.retrieval_query,
                    profile,
                    rewrite_result=plan.rewrite_result,
                    progress=progress,
                )
        addresses = [result.address for result in read_results]
        retrieval_duration = time.perf_counter() - t0
        if not use_multi_hop:
            timings.extend(_retrieval_pass_timings(self._retrieval_pass))
        timings.append(("Retrieval", retrieval_duration))

        if not read_results:
            return RetrievalOutcome(
                sanitized=sanitized,
                expanded=[],
                addresses=[],
                timings=timings,
                profile=profile,
                retrieval_query=plan.retrieval_query,
                rewrite_result=plan.rewrite_result,
                query_profile_metadata=query_profile,
            )

        if expand_context:
            t0 = time.perf_counter()
            expanded = self._expander.expand(
                read_results,
                entity_expansion_limit=profile.entity_expansion_limit,
            )
            timings.append(("Expand context", time.perf_counter() - t0))
        else:
            expanded = read_results

        if execute_table_queries:
            t0 = time.perf_counter()
            expanded = self._table_handler.process(sanitized, expanded)
            timings.append(("Table queries", time.perf_counter() - t0))

        return RetrievalOutcome(
            sanitized=sanitized,
            expanded=expanded,
            addresses=addresses,
            timings=timings,
            profile=profile,
            retrieval_query=plan.retrieval_query,
            rewrite_result=plan.rewrite_result,
            query_profile_metadata=query_profile,
        )

    def retry_retrieve(
        self,
        outcome: RetrievalOutcome,
        *,
        retrieval_action: str,
        retrieval_modality: str | None = None,
        progress: Callable[[str], None] | None = None,
        allow_llm_strategies: bool = True,
        execute_table_queries: bool = True,
        expand_context: bool = True,
    ) -> RetrievalOutcome:
        """Run one Pyrrho-directed follow-up retrieval pass and merge new evidence."""
        if outcome.profile is None:
            return outcome

        profile = _retry_profile(
            outcome.profile,
            self._config,
            retrieval_action=retrieval_action,
            retrieval_modality=retrieval_modality,
        )
        query = outcome.retrieval_query or outcome.sanitized
        existing_keys = _read_result_keys(outcome.expanded)
        timings = list(outcome.timings)

        _progress = progress or (lambda _: None)
        _progress("Retrieving additional evidence...")
        t0 = time.perf_counter()
        with self._retrieval_strategy_scope(allow_llm_strategies):
            read_results = self._retrieval_pass.run(
                query,
                profile,
                exclude=existing_keys,
                rewrite_result=outcome.rewrite_result,
                progress=progress,
            )
        retry_duration = time.perf_counter() - t0
        timings.extend(_prefixed_retrieval_pass_timings(self._retrieval_pass, "Retry "))
        timings.append(("Retrieval retry", retry_duration))

        if expand_context and read_results:
            t0 = time.perf_counter()
            read_results = self._expander.expand(
                read_results,
                entity_expansion_limit=profile.entity_expansion_limit,
            )
            timings.append(("Retry expand context", time.perf_counter() - t0))

        if execute_table_queries and read_results:
            t0 = time.perf_counter()
            read_results = self._table_handler.process(outcome.sanitized, read_results)
            timings.append(("Retry table queries", time.perf_counter() - t0))

        expanded = _merge_read_results(outcome.expanded, read_results)
        return RetrievalOutcome(
            sanitized=outcome.sanitized,
            expanded=expanded,
            addresses=[result.address for result in expanded],
            timings=timings,
            profile=profile,
            retrieval_query=query,
            rewrite_result=outcome.rewrite_result,
            query_profile_metadata=outcome.query_profile_metadata,
        )

    def _prepare_query_plan(
        self,
        sanitized: str,
        metadata: dict[str, Any],
        *,
        use_query_intelligence: bool | None,
    ) -> QueryPlan:
        """Build the deterministic plan, optionally enhanced by query intelligence."""
        if use_query_intelligence is None:
            use_query_intelligence = self._config.query_intelligence is not None

        planner = self._query_planner or DeterministicQueryPlanner()
        plan = planner.plan(sanitized, detection_enabled=True)

        if not use_query_intelligence:
            return plan

        fast_analysis = self._fast_analyze(sanitized)
        need_llm_analysis = fast_analysis is None
        need_detection = self._needs_detection(sanitized)

        try:
            batch_result = self._query_batcher.batch_classify(
                sanitized,
                include_analysis=need_llm_analysis,
                include_detection=need_detection,
                include_rewriting=True,
                include_extended=True,
                include_keywords=True,
                conversation_context=metadata.get("conversation_context"),
            )
            llm_detection = (
                self._build_detection_summary(batch_result.detection_results)
                if need_detection and batch_result.detection_results is not None
                else plan.detection
            )
            plan = plan_from_batch_result(
                sanitized,
                batch_result,
                fallback_analysis=fast_analysis or plan.analysis,
                detection=llm_detection,
                fallback_plan=plan,
            )
            if plan.rewrite_result and plan.retrieval_query != sanitized:
                logger.debug(
                    "Query rewritten",
                    original_preview=sanitized[:50],
                    rewritten_preview=plan.retrieval_query[:50],
                )
        except Exception as e:
            logger.warning(f"Batched query intelligence failed: {e}")

        return plan

    def _classify_query_signals(self, sanitized: str) -> Any:
        """Classify Pyrrho query-planning signals when the backend exposes them."""
        classifier = self._query_signal_classifier
        classify_query = getattr(classifier, "classify_query", None)
        if not callable(classify_query) or _is_mock_callable(classify_query):
            return None
        return classify_query(sanitized)

    def _add_semantic_query_keywords(self, query: str, plan: QueryPlan) -> QueryPlan:
        """Use local Qwen for keyword-only query expansion."""
        batcher = self._semantic_keyword_batcher
        if batcher is None:
            return plan

        try:
            batch_result = batcher.batch_classify(
                query,
                include_analysis=False,
                include_detection=False,
                include_rewriting=False,
                include_extended=False,
                include_keywords=True,
            )
        except Exception as e:
            logger.debug(f"Semantic query keyword expansion failed: {e}")
            return plan

        if not batch_result.keywords:
            return plan

        return QueryPlan(
            retrieval_query=plan.retrieval_query,
            analysis=plan.analysis,
            detection=plan.detection,
            rewrite_result=plan.rewrite_result,
            extended_signals=plan.extended_signals,
            keywords=_merge_query_keywords(plan.keywords, batch_result.keywords),
        )


def _sanitize_query(text: str) -> str:
    """Strip tags and cap pathologically long input."""
    sanitized = re.sub(r"<[^>]+>", "", text).strip()
    if not sanitized:
        sanitized = text.strip()

    if len(sanitized) > _MAX_QUERY_LENGTH:
        original_length = len(sanitized)
        sanitized = sanitized[:_MAX_QUERY_LENGTH]
        logger.debug(
            "Query truncated",
            original_length=original_length,
            new_length=_MAX_QUERY_LENGTH,
        )
    return sanitized


def _retrieval_pass_timings(retrieval_pass: Any) -> list[tuple[str, float]]:
    """Read the latest one-pass timing breakdown."""
    pass_timings = getattr(retrieval_pass, "last_timings", {})
    return [
        (label, pass_timings[key])
        for key, label in (
            ("recall", "Recall"),
            ("rerank", "Rerank"),
            ("read", "Read"),
        )
        if key in pass_timings
    ]


def _prefixed_retrieval_pass_timings(retrieval_pass: Any, prefix: str) -> list[tuple[str, float]]:
    """Read one-pass timings with a label prefix."""
    return [
        (f"{prefix}{name}", duration) for name, duration in _retrieval_pass_timings(retrieval_pass)
    ]


def _retry_profile(
    profile: Any,
    config: "FitzKragConfig",
    *,
    retrieval_action: str,
    retrieval_modality: str | None,
) -> Any:
    """Build a broader profile for one Pyrrho-directed retry pass."""
    weights = dict(getattr(profile, "strategy_weights", {}) or {})
    top_k = max(int(getattr(profile, "top_k", config.top_addresses)), config.top_addresses)
    top_read = max(int(getattr(profile, "top_read", config.top_read)), config.top_read)
    kwargs: dict[str, Any] = {
        "strategy_weights": weights,
        "top_k": int(top_k * 2),
        "top_read": int(top_read * 1.5),
        "run_agentic": True,
    }

    if retrieval_action == "broaden_search":
        weights.update(
            {
                "code": max(weights.get("code", 0.0), 0.25),
                "section": max(weights.get("section", 0.0), 0.35),
                "table": max(weights.get("table", 0.0), 0.20),
            }
        )
        kwargs.update(
            {
                "specificity": "broad",
                "answer_type": "exploratory",
                "inject_corpus_summaries": True,
                "entity_expansion_limit": max(
                    int(getattr(profile, "entity_expansion_limit", 3)),
                    12,
                ),
            }
        )
    elif retrieval_action == "resolve_conflict":
        kwargs.update(
            {
                "has_comparison_intent": True,
                "answer_type": "comparative",
            }
        )
    elif retrieval_action == "structured_lookup":
        weights["table"] = max(weights.get("table", 0.0), 0.45)
        kwargs.update(
            {
                "query_contract": "structured_lookup",
                "answer_type": "factual",
            }
        )

    apply_retrieval_modality_weights(weights, retrieval_modality)
    return replace(profile, **kwargs)


def _read_result_keys(results: list["ReadResult"]) -> set[tuple[str, str]]:
    """Return address keys for already-read evidence."""
    return {
        (str(result.address.source_id), str(result.address.location))
        for result in results
        if getattr(result, "address", None) is not None
    }


def _merge_read_results(
    current: list["ReadResult"],
    additional: list["ReadResult"],
) -> list["ReadResult"]:
    """Merge read results by address key, preserving the original order."""
    merged = list(current)
    seen = _read_result_keys(merged)
    for result in additional:
        address = getattr(result, "address", None)
        if address is None:
            merged.append(result)
            continue
        key = (str(address.source_id), str(address.location))
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
    return merged


def _merge_query_keywords(*keyword_lists: list[str]) -> list[str]:
    """Merge query keyword lists while preserving first occurrence."""
    merged: list[str] = []
    seen: set[str] = set()
    for keywords in keyword_lists:
        for keyword in keywords:
            value = str(keyword).strip()
            key = value.lower()
            if value and key not in seen:
                seen.add(key)
                merged.append(value)
    return merged


def _is_mock_callable(value: Any) -> bool:
    """Return whether a callable came from unittest.mock."""
    if isinstance(value, Mock):
        return True
    return isinstance(getattr(value, "__self__", None), Mock)


__all__ = ["QueryPipeline", "RetrievalOutcome"]

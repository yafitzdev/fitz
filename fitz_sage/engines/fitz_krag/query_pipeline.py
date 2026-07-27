# fitz_sage/engines/fitz_krag/query_pipeline.py
"""Query-side retrieval pipeline for KRAG."""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.evidence_closure import (
    EvidenceClosureRequest,
    annotate_closure_result,
    plan_evidence_closure,
    request_metadata,
)
from fitz_sage.engines.fitz_krag.evidence_compiler import compile_evidence
from fitz_sage.engines.fitz_krag.evidence_contract import build_query_contract
from fitz_sage.engines.fitz_krag.query_planner import (
    DeterministicQueryPlanner,
    QueryPlan,
    content_terms,
    plan_from_batch_result,
)
from fitz_sage.engines.fitz_krag.retrieval.trace import read_results_trace
from fitz_sage.engines.fitz_krag.retrieval_profile import (
    RetrievalProfile,
    build_retrieval_profile,
    query_profile_metadata,
)
from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.core import Query
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.evidence_compiler import EvidenceCompilation
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
    retrieval_trace: dict[str, Any] = field(default_factory=dict)
    query_terms: list[dict[str, str]] = field(default_factory=list)


class QueryPipeline:
    """Analyze, retrieve, read, expand, and table-process one query."""

    def __init__(
        self,
        *,
        config: "FitzKragConfig",
        query_planner: Any,
        query_batcher: Any,
        semantic_keyword_batcher: Any,
        pyrrho: Any,
        retrieval_pass: Any,
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
        self._semantic_keyword_batcher = semantic_keyword_batcher
        self._pyrrho = pyrrho
        self._retrieval_pass = retrieval_pass
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
        allow_table_sql_generation: bool = True,
        expand_context: bool = True,
    ) -> RetrievalOutcome:
        """Run the retrieval half of the KRAG pipeline."""
        sanitized = _sanitize_query(query.text)
        timings: list[tuple[str, float]] = []

        _progress = progress or (lambda _: None)
        _progress("Analyzing query...")

        t0 = time.perf_counter()
        prepared_plan, prepared_keyword_origin = self._prepare_query_plan(
            sanitized,
            query.metadata,
            use_query_intelligence=use_query_intelligence,
        )
        timings.append(("Query prep", time.perf_counter() - t0))

        t0 = time.perf_counter()
        plan = self._add_semantic_query_keywords(sanitized, prepared_plan)
        timings.append(("Qwen query keywords", time.perf_counter() - t0))
        query_terms = _query_term_trace(
            sanitized,
            prepared_plan,
            plan,
            prepared_keyword_origin=prepared_keyword_origin,
        )

        t0 = time.perf_counter()
        pyrrho_plan = self._plan_with_pyrrho(sanitized)
        timings.append(("Pyrrho pre", time.perf_counter() - t0))

        profile = build_retrieval_profile(
            plan.analysis,
            plan.detection,
            self._config,
            extended_signals=plan.extended_signals,
            keywords=plan.keywords,
            pyrrho_plan=pyrrho_plan,
        )
        query_profile = query_profile_metadata(profile, pyrrho_plan)

        _progress("Retrieving relevant sources...")
        t0 = time.perf_counter()
        with self._retrieval_strategy_scope(allow_llm_strategies):
            read_results = self._retrieval_pass.run(
                plan.retrieval_query,
                profile,
                rewrite_result=plan.rewrite_result,
                progress=progress,
            )
            retrieval_trace = dict(getattr(self._retrieval_pass, "last_trace", {}) or {})
        addresses = [result.address for result in read_results]
        retrieval_duration = time.perf_counter() - t0
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
                retrieval_trace=retrieval_trace,
                query_terms=query_terms,
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
            expanded = self._table_handler.process(
                sanitized,
                expanded,
                allow_sql_generation=allow_table_sql_generation,
            )
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
            retrieval_trace=retrieval_trace,
            query_terms=query_terms,
        )

    def close_evidence(
        self,
        outcome: RetrievalOutcome,
        compilation: "EvidenceCompilation",
        *,
        progress: Callable[[str], None] | None = None,
        allow_llm_strategies: bool = True,
        execute_table_queries: bool = True,
        allow_table_sql_generation: bool = True,
        expand_context: bool = True,
    ) -> RetrievalOutcome:
        """Run contract-driven follow-up retrieval for unresolved evidence obligations."""
        plan = plan_evidence_closure(
            outcome.sanitized,
            outcome.expanded,
            compilation,
            profile=outcome.profile,
        )
        retrieval_trace = dict(outcome.retrieval_trace)
        closure_trace = dict(plan.metadata)
        closure_trace["runs"] = []
        if not plan.requests:
            retrieval_trace["evidence_closure"] = closure_trace
            return replace(outcome, retrieval_trace=retrieval_trace)

        contract = build_query_contract(outcome.sanitized, outcome.profile)
        expanded = list(outcome.expanded)
        closure_evidence = list(compilation.results)
        timings = list(outcome.timings)
        total_added = 0
        total_replaced = 0
        _progress = progress or (lambda _: None)

        for run_index, request in enumerate(plan.requests, start=1):
            _progress("Closing evidence obligations...")
            profile = _closure_profile(outcome.profile, self._config, request)

            t0 = time.perf_counter()
            with self._retrieval_strategy_scope(allow_llm_strategies):
                read_results = self._retrieval_pass.run(
                    request.query,
                    profile,
                    rewrite_result=None,
                    progress=progress,
                )
            retrieved_count = len(read_results)
            read_results = _filter_companion_source_repeats(
                request,
                closure_evidence,
                read_results,
            )
            closure_duration = time.perf_counter() - t0
            timings.extend(
                _prefixed_retrieval_pass_timings(
                    self._retrieval_pass,
                    f"Evidence closure {run_index} ",
                )
            )
            timings.append((f"Evidence closure {run_index}", closure_duration))

            retrieval_pass_trace = dict(getattr(self._retrieval_pass, "last_trace", {}) or {})
            if expand_context and read_results:
                t0 = time.perf_counter()
                read_results = self._expander.expand(
                    read_results,
                    entity_expansion_limit=profile.entity_expansion_limit,
                )
                timings.append((f"Evidence closure {run_index} expand", time.perf_counter() - t0))

            if execute_table_queries and read_results:
                t0 = time.perf_counter()
                read_results = self._table_handler.process(
                    request.query,
                    read_results,
                    allow_sql_generation=allow_table_sql_generation,
                )
                timings.append(
                    (f"Evidence closure {run_index} table queries", time.perf_counter() - t0)
                )

            selected_results = _select_closure_results(
                request.query,
                read_results,
                profile,
                request=request,
            )
            annotated = [
                annotate_closure_result(
                    result,
                    request,
                    contract=contract,
                    run_index=run_index,
                )
                for result in selected_results
            ]
            closure_evidence.extend(annotated)
            expanded, added, replaced = _merge_closure_results(
                expanded,
                annotated,
                allow_replace=True,
            )
            total_added += added
            total_replaced += replaced
            closure_trace["runs"].append(
                {
                    "request": request_metadata(request),
                    "trace": retrieval_pass_trace,
                    "retrieved_count": retrieved_count,
                    "read_count": len(read_results),
                    "selected_count": len(selected_results),
                    "added": added,
                    "replaced": replaced,
                    "results": read_results_trace(annotated),
                }
            )

        closure_trace["added"] = total_added
        closure_trace["replaced"] = total_replaced
        retrieval_trace["evidence_closure"] = closure_trace
        return RetrievalOutcome(
            sanitized=outcome.sanitized,
            expanded=expanded,
            addresses=[result.address for result in expanded],
            timings=timings,
            profile=outcome.profile,
            retrieval_query=outcome.retrieval_query,
            rewrite_result=outcome.rewrite_result,
            query_profile_metadata=outcome.query_profile_metadata,
            retrieval_trace=retrieval_trace,
            query_terms=outcome.query_terms,
        )

    def _prepare_query_plan(
        self,
        sanitized: str,
        metadata: dict[str, Any],
        *,
        use_query_intelligence: bool | None,
    ) -> tuple[QueryPlan, str]:
        """Build the deterministic plan, optionally enhanced by query intelligence."""
        if use_query_intelligence is None:
            use_query_intelligence = self._config.query_intelligence is not None

        planner = self._query_planner or DeterministicQueryPlanner()
        plan = planner.plan(sanitized, detection_enabled=True)

        if not use_query_intelligence:
            return plan, "deterministic"

        fast_analysis = self._fast_analyze(sanitized)
        need_llm_analysis = fast_analysis is None
        need_detection = self._needs_detection(sanitized)

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
        keyword_origin = "query_intelligence" if batch_result.keywords else "deterministic"
        plan = plan_from_batch_result(
            sanitized,
            batch_result,
            fallback_analysis=fast_analysis or plan.analysis,
            detection=llm_detection,
            fallback_plan=plan,
        )
        if plan.rewrite_result and plan.retrieval_query != sanitized:
            logger.debug(
                "Query rewritten: %r -> %r",
                sanitized[:50],
                plan.retrieval_query[:50],
            )

        return plan, keyword_origin

    def _add_semantic_query_keywords(self, query: str, plan: QueryPlan) -> QueryPlan:
        """Use local Qwen for keyword-only query expansion."""
        batcher = self._semantic_keyword_batcher
        if batcher is None:
            return plan

        batch_result = batcher.batch_classify(
            query,
            include_analysis=False,
            include_detection=False,
            include_rewriting=False,
            include_extended=False,
            include_keywords=True,
        )

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

    def _plan_with_pyrrho(self, query: str) -> Any | None:
        """Run Pyrrho's query-only planning pass when the backend exposes it."""
        planner = getattr(self._pyrrho, "plan_query", None)
        if not callable(planner):
            return None
        return planner(query)


def _sanitize_query(text: str) -> str:
    """Strip tags and cap pathologically long input."""
    sanitized = re.sub(r"<[^>]+>", "", text).strip()
    if not sanitized:
        sanitized = text.strip()

    if len(sanitized) > _MAX_QUERY_LENGTH:
        original_length = len(sanitized)
        sanitized = sanitized[:_MAX_QUERY_LENGTH]
        logger.debug("Query truncated: %d -> %d characters", original_length, _MAX_QUERY_LENGTH)
    return sanitized


def _query_term_trace(
    query: str,
    prepared_plan: QueryPlan,
    expanded_plan: QueryPlan,
    *,
    prepared_keyword_origin: str,
) -> list[dict[str, str]]:
    """Record term provenance without exposing planner implementation objects."""
    traced: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    def add(values: list[str], origin: str) -> None:
        for value in values:
            text = str(value).strip()
            key = (text.casefold(), origin)
            if not text or key in seen:
                continue
            seen.add(key)
            traced.append({"text": text, "origin": origin})

    add(content_terms(query), "literal")
    add(list(prepared_plan.keywords), prepared_keyword_origin)
    prepared = {keyword.casefold() for keyword in prepared_plan.keywords}
    add(
        [keyword for keyword in expanded_plan.keywords if keyword.casefold() not in prepared],
        "semantic",
    )
    return traced


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


def _closure_profile(
    profile: Any,
    config: "FitzKragConfig",
    request: EvidenceClosureRequest,
) -> RetrievalProfile:
    """Build a tight executor profile for one evidence-contract closure request."""
    base = profile or RetrievalProfile(
        top_k=config.top_addresses,
        top_read=config.top_read,
    )
    top_k = max(12, min(int(getattr(base, "top_k", config.top_addresses)), 32))
    top_read = max(6, min(int(getattr(base, "top_read", config.top_read)), 12))
    weights = {
        "code": 0.01,
        "section": 0.01,
        "table": 0.01,
    }
    if request.modality == "table":
        weights.update({"table": 1.0, "section": 0.12})
    elif request.modality == "symbol":
        weights.update({"code": 1.0, "section": 0.12})
    else:
        weights.update({"section": 1.0, "code": 0.04, "table": 0.04})

    return replace(
        base,
        strategy_weights=weights,
        top_k=top_k,
        top_read=top_read,
        query_contract=getattr(base, "query_contract", None),
        retrieval_modality=getattr(base, "retrieval_modality", None),
        retrieval_obligation=getattr(base, "retrieval_obligation", None),
        required_modalities=(request.modality,),
        run_agentic=False,
        inject_corpus_summaries=False,
        entity_expansion_limit=min(int(getattr(base, "entity_expansion_limit", 3)), 3),
        specificity=getattr(base, "specificity", "moderate"),
        answer_type=getattr(base, "answer_type", "factual"),
    )


def _merge_closure_results(
    current: list["ReadResult"],
    additional: list["ReadResult"],
    *,
    allow_replace: bool,
) -> tuple[list["ReadResult"], int, int]:
    """Merge closure results while allowing bridge-grounded table refreshes."""
    merged = list(current)
    positions = {
        (str(result.address.source_id), str(result.address.location)): index
        for index, result in enumerate(merged)
        if getattr(result, "address", None) is not None
    }
    added = 0
    replaced = 0
    for result in additional:
        address = getattr(result, "address", None)
        if address is None:
            merged.append(result)
            added += 1
            continue
        key = (str(address.source_id), str(address.location))
        if key in positions:
            if allow_replace and _closure_result_should_replace(merged[positions[key]], result):
                merged[positions[key]] = result
                replaced += 1
            continue
        positions[key] = len(merged)
        merged.append(result)
        added += 1
    return merged, added, replaced


def _select_closure_results(
    query: str,
    results: list["ReadResult"],
    profile: Any,
    *,
    request: EvidenceClosureRequest | None = None,
) -> list["ReadResult"]:
    """Keep the best grounded result for one bounded closure obligation."""
    if not results:
        return []
    if request is not None and request.role.startswith("bridge_document:"):
        document = _normalize_source_name(request.role.removeprefix("bridge_document:"))
        exact_locations = [
            result
            for result in results
            if _normalize_source_name(str(result.address.location)) == document
        ]
        if exact_locations:
            results = exact_locations
    elif request is not None and request.role.startswith("bridge_definition:"):
        definition = _normalize_source_name(request.role.removeprefix("bridge_definition:"))
        definition_matches = [
            result
            for result in results
            if definition in _normalize_source_name(f"{result.address.location} {result.content}")
        ]
        if definition_matches:
            results = definition_matches
    compilation = compile_evidence(query, results, profile=profile)
    if compilation.results:
        return compilation.results[:1]
    return results[:1]


def _normalize_source_name(value: str) -> str:
    """Normalize a source-derived document label for exact companion matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _filter_companion_source_repeats(
    request: EvidenceClosureRequest,
    existing: list["ReadResult"],
    candidates: list["ReadResult"],
) -> list["ReadResult"]:
    """Keep non-table companion retrieval on sources not already in evidence."""
    if request.modality == "table":
        return candidates
    existing_sources = {
        str(result.address.source_id)
        for result in existing
        if getattr(result, "address", None) is not None
    }
    return [
        result for result in candidates if str(result.address.source_id) not in existing_sources
    ]


def _closure_result_should_replace(existing: "ReadResult", candidate: "ReadResult") -> bool:
    """Return whether closure produced a more specific version of an existing result."""
    candidate_closure = candidate.metadata.get("evidence_closure")
    existing_closure = existing.metadata.get("evidence_closure")
    candidate_deterministic = bool(candidate.metadata.get("deterministic_table_filter"))
    existing_deterministic = bool(existing.metadata.get("deterministic_table_filter"))
    if candidate_deterministic and not existing_deterministic:
        return True
    if existing_deterministic and not candidate_deterministic:
        return False

    candidate_count = _metadata_int(candidate.metadata.get("result_count"))
    existing_count = _metadata_int(existing.metadata.get("result_count"))
    if candidate_count > 0 and existing_count > 0 and candidate_count != existing_count:
        return candidate_count < existing_count
    if candidate_count > 0 and existing_count == 0:
        return True
    if existing_count > 0 and candidate_count == 0 and existing_deterministic:
        return False

    if isinstance(candidate_closure, dict) and not isinstance(existing_closure, dict):
        return True
    return False


def _metadata_int(value: Any) -> int:
    """Return an integer metadata value, or zero when absent."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


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


__all__ = ["QueryPipeline", "RetrievalOutcome"]

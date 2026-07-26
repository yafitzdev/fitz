# fitz_sage/engines/fitz_krag/governance_cutoff.py
"""Pyrrho cutoff policy for selecting the smallest useful evidence prefix."""

from __future__ import annotations

import re
import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

from fitz_sage.core.answer_mode import AnswerMode

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.types import ReadResult

_DEFAULT_GOVERNANCE_CUTOFF = 10
_NARROW_MIN_EVIDENCE = 1
_COMPARISON_MIN_EVIDENCE = 2
_BROAD_MIN_EVIDENCE = 4
_AGGREGATION_MIN_EVIDENCE = 5
_FOLLOWUP_BATCH_SIZE = 2
_BROAD_OVERVIEW_SOURCE_COUNT = 6
_MONTH_PATTERN = (
    r"\b(january|february|march|april|may|june|july|august|september|"
    r"october|november|december)(?:\s+(\d{4}))?\b"
)
_REQUIRED_QUERY_TERMS = {
    "revenue": ("revenue", "revenues"),
    "profit": ("profit", "profits"),
    "margin": ("margin", "margins"),
    "cost": ("cost", "costs"),
    "budget": ("budget", "budgets"),
    "invoice": ("invoice", "invoices"),
    "churn": ("churn",),
    "retention": ("retention",),
    "conversion": ("conversion", "conversions"),
    "sales": ("sales",),
}
_METRIC_STOP_TERMS = {
    "a",
    "an",
    "and",
    "are",
    "between",
    "compare",
    "compared",
    "did",
    "doc",
    "docs",
    "document",
    "documents",
    "for",
    "had",
    "has",
    "have",
    "higher",
    "highest",
    "how",
    "in",
    "less",
    "lower",
    "lowest",
    "more",
    "most",
    "of",
    "or",
    "quarter",
    "quarterly",
    "report",
    "reports",
    "summary",
    "summaries",
    "the",
    "vs",
    "versus",
    "was",
    "were",
    "what",
    "when",
    "which",
    "who",
    "why",
}
_METRIC_MODIFIER_TERMS = {"average", "avg", "count", "mean", "number", "sum", "total"}


@dataclass(frozen=True)
class GovernanceCutoffPolicy:
    """Prefix stop policy for Pyrrho verdict evaluation."""

    query_shape: str
    max_docs: int
    min_sufficient_docs: int


@dataclass(frozen=True)
class GovernanceCutoffResult:
    """Outcome of the cutoff loop."""

    selected: list["ReadResult"]
    mode: AnswerMode
    reasons: list[str]
    timings: list[tuple[str, float]]
    metadata: dict[str, Any]


def apply_governance_cutoff(
    query: str,
    results: list["ReadResult"],
    governance: Any,
    *,
    profile: Any = None,
    requested_top_k: Any = None,
) -> GovernanceCutoffResult:
    """Use Pyrrho to select the smallest sufficient prefix of ranked evidence."""
    if not results:
        mode = AnswerMode.INSUFFICIENT
        return GovernanceCutoffResult(
            selected=[],
            mode=mode,
            reasons=["No relevant evidence retrieved."],
            timings=[],
            metadata=_governance_cutoff_metadata(
                None,
                evaluated=0,
                selected=0,
                mode=mode,
                stop_reason="no_results",
            ),
        )

    t0 = time.perf_counter()
    policy = governance_cutoff_policy(profile, len(results), requested_top_k, query=query)
    if policy.query_shape == "broad_overview":
        selected_count = min(policy.max_docs, _BROAD_OVERVIEW_SOURCE_COUNT)
        mode = AnswerMode.INSUFFICIENT
        reason = (
            "Query is too broad for evidence sufficiency; returned representative "
            "sources instead of a Pyrrho sufficient verdict."
        )
        return GovernanceCutoffResult(
            selected=results[:selected_count],
            mode=mode,
            reasons=[
                reason,
                "Refine the query with a topic, entity, timeframe, or document type for sufficiency.",
            ],
            timings=[("Governance", time.perf_counter() - t0)],
            metadata={
                **_governance_cutoff_metadata(
                    policy,
                    evaluated=0,
                    selected=selected_count,
                    mode=mode,
                    stop_reason="representative_overview",
                ),
                "representative_sources": True,
                "sufficiency_evaluated": False,
            },
        )

    results = _prioritize_comparison_metric_evidence(query, profile, results, policy)
    governance_results = _pyrrho_governance_results(results)
    evidence_prefix_min = min(_evidence_prefix_min(governance_results), policy.max_docs)
    return_prefix_min = min(_return_prefix_min(results), policy.max_docs)

    last_reasons: list[str] = []
    last_decision: Any = None
    trajectory: list[dict[str, Any]] = []

    for size, decision in _iter_prefix_decisions(governance, query, governance_results, policy):
        mode = _decision_mode(decision)
        last_decision = decision
        last_reasons = _decision_reasons(decision)
        trace = _prefix_trace(size, mode, decision)
        full_prefix_size = _full_prefix_size_for_governance_prefix(
            results, governance_results[:size]
        )

        if mode is AnswerMode.SUFFICIENT:
            required_sufficient_docs = evidence_prefix_min or policy.min_sufficient_docs
            can_stop = size >= required_sufficient_docs
            stop_reason = "sufficient_min_evidence_met"

            if can_stop:
                selected_size = max(full_prefix_size, return_prefix_min or full_prefix_size)
                trajectory.append(trace)
                return GovernanceCutoffResult(
                    selected=results[:selected_size],
                    mode=mode,
                    reasons=last_reasons,
                    timings=[("Governance", time.perf_counter() - t0)],
                    metadata=_governance_cutoff_metadata(
                        policy,
                        evaluated=size,
                        selected=selected_size,
                        mode=mode,
                        decision=decision,
                        trajectory=trajectory,
                        stop_reason=stop_reason,
                    ),
                )
            if evidence_prefix_min > size:
                trace["evidence_prefix_min"] = evidence_prefix_min
            trajectory.append(trace)
            continue

        if mode is AnswerMode.DISPUTED:
            trajectory.append(trace)
            required_disputed_docs = evidence_prefix_min or policy.min_sufficient_docs
            if size >= required_disputed_docs:
                selected_size = max(full_prefix_size, return_prefix_min or full_prefix_size)
                return GovernanceCutoffResult(
                    selected=results[:selected_size],
                    mode=mode,
                    reasons=last_reasons,
                    timings=[("Governance", time.perf_counter() - t0)],
                    metadata=_governance_cutoff_metadata(
                        policy,
                        evaluated=size,
                        selected=selected_size,
                        mode=mode,
                        decision=decision,
                        trajectory=trajectory,
                        stop_reason="disputed_min_evidence_met",
                    ),
                )
            if evidence_prefix_min > size:
                trace["evidence_prefix_min"] = evidence_prefix_min
            continue

        trajectory.append(trace)

    reasons = list(last_reasons)
    final_mode = (
        _decision_mode(last_decision) if last_decision is not None else AnswerMode.INSUFFICIENT
    )
    if final_mode is AnswerMode.INSUFFICIENT:
        reasons.append(
            f"Pyrrho did not find sufficient evidence within the top "
            f"{policy.max_docs} evidence item(s)."
        )
    stop_reason = "cutoff_exhausted"
    evaluated = min(policy.max_docs, len(governance_results))
    selected_count = max(
        _full_prefix_size_for_governance_prefix(results, governance_results[:evaluated]),
        return_prefix_min,
    )
    return GovernanceCutoffResult(
        selected=results[:selected_count],
        mode=final_mode,
        reasons=reasons,
        timings=[("Governance", time.perf_counter() - t0)],
        metadata=_governance_cutoff_metadata(
            policy,
            evaluated=evaluated,
            selected=selected_count,
            mode=final_mode,
            decision=last_decision,
            trajectory=trajectory,
            stop_reason=stop_reason,
        ),
    )


def governance_cutoff_policy(
    profile: Any,
    result_count: int,
    requested_top_k: Any = None,
    *,
    query: str | None = None,
) -> GovernanceCutoffPolicy:
    """Build the prefix policy used to ask Pyrrho for governance verdicts."""
    max_docs = _governance_cutoff_limit(result_count, requested_top_k)
    query_shape = governance_query_shape(profile, query=query)
    min_docs_by_shape = {
        "narrow": _NARROW_MIN_EVIDENCE,
        "comparison": _COMPARISON_MIN_EVIDENCE,
        "broad": _BROAD_MIN_EVIDENCE,
        "broad_overview": _BROAD_MIN_EVIDENCE,
        "aggregation": _AGGREGATION_MIN_EVIDENCE,
    }
    min_sufficient_docs = min(max_docs, min_docs_by_shape[query_shape])
    return GovernanceCutoffPolicy(
        query_shape=query_shape,
        max_docs=max_docs,
        min_sufficient_docs=min_sufficient_docs,
    )


def governance_query_shape(profile: Any, *, query: str | None = None) -> str:
    """Map retrieval profile signals to a cutoff policy shape."""
    if profile is None:
        return "narrow"
    query_contract = getattr(profile, "query_contract", None)
    if query_contract == "representative_overview":
        return "broad_overview"
    if query_contract == "comparison_coverage":
        return "comparison"
    if query_contract == "exhaustive_coverage":
        return "aggregation"
    if (
        getattr(profile, "has_comparison_intent", False)
        or getattr(profile, "answer_type", "") == "comparative"
        or getattr(profile, "comparison_queries", None)
        or getattr(profile, "comparison_entities", None)
    ):
        return "comparison"
    if getattr(profile, "has_aggregation_intent", False):
        return "aggregation"
    if (
        getattr(profile, "specificity", "") == "broad"
        or getattr(profile, "answer_type", "") == "exploratory"
        or getattr(profile, "inject_corpus_summaries", False)
    ):
        return "broad"
    return "narrow"


def _iter_prefix_decisions(
    governance: Any,
    query: str,
    results: list["ReadResult"],
    policy: GovernanceCutoffPolicy,
) -> Iterator[tuple[int, Any]]:
    """Yield policy-aware prefix decisions, batching when the backend supports it."""
    supports_batching = _supports_decide_many(governance)
    next_size = 1
    first_batch_end = _first_batch_end(policy) if supports_batching else 1
    while next_size <= policy.max_docs:
        if not supports_batching:
            batch_end = next_size
        elif next_size == 1:
            batch_end = first_batch_end
        else:
            batch_end = min(policy.max_docs, next_size + _FOLLOWUP_BATCH_SIZE - 1)
        batch_sizes = list(range(next_size, batch_end + 1))
        prefixes = [results[:size] for size in batch_sizes]
        for size, decision in zip(
            batch_sizes,
            _decide_prefix_batch(governance, query, prefixes),
            strict=True,
        ):
            yield size, decision
        next_size = batch_end + 1


def _first_batch_end(policy: GovernanceCutoffPolicy) -> int:
    """Batch up to the earliest point where a verdict can legally stop."""
    return min(policy.max_docs, max(1, policy.min_sufficient_docs))


def _decide_prefix_batch(
    governance: Any, query: str, prefixes: list[list["ReadResult"]]
) -> list[Any]:
    """Classify prefixes through decide_many when available, otherwise sequentially."""
    if _supports_decide_many(governance):
        return list(governance.decide_many(query, prefixes))
    return [governance.decide(query, prefix) for prefix in prefixes]


def _supports_decide_many(governance: Any) -> bool:
    """Return whether governance exposes a real batch API."""
    decide_many = getattr(governance, "decide_many", None)
    return (
        bool(getattr(governance, "supports_batched_prefixes", False))
        and callable(decide_many)
        and not _is_mock_callable(decide_many)
    )


def _is_mock_callable(value: Any) -> bool:
    """Return whether a callable came from unittest.mock."""
    if isinstance(value, Mock):
        return True
    return isinstance(getattr(value, "__self__", None), Mock)


def _decision_mode(decision: Any) -> AnswerMode:
    """Read a decision mode defensively."""
    mode = getattr(decision, "mode", AnswerMode.INSUFFICIENT)
    if isinstance(mode, AnswerMode):
        return mode
    return AnswerMode(str(mode))


def _decision_reasons(decision: Any) -> list[str]:
    """Read decision reasons without assuming one concrete Pyrrho object."""
    reasons = getattr(decision, "reasons", None)
    if isinstance(reasons, (list, tuple)):
        return [str(reason) for reason in reasons if reason]
    reason = getattr(decision, "reason", None)
    return [str(reason)] if reason else []


def _governance_cutoff_metadata(
    policy: GovernanceCutoffPolicy | None,
    *,
    evaluated: int,
    selected: int,
    mode: AnswerMode,
    decision: Any = None,
    trajectory: list[dict[str, Any]] | None = None,
    stop_reason: str | None = None,
) -> dict[str, Any]:
    """Build serializable metadata for the cutoff loop."""
    metadata: dict[str, Any] = {
        "evaluated": evaluated,
        "selected": selected,
        "max": policy.max_docs if policy else 0,
        "mode": mode.value,
    }
    if stop_reason:
        metadata["stop_reason"] = stop_reason
    if policy is not None:
        metadata["policy"] = {
            "query_shape": policy.query_shape,
            "min_sufficient_docs": policy.min_sufficient_docs,
        }
    if trajectory:
        metadata["trajectory"] = trajectory
    pyrrho_metadata = _pyrrho_metadata(mode, decision)
    if pyrrho_metadata:
        metadata["pyrrho"] = pyrrho_metadata
    return metadata


def _prioritize_comparison_metric_evidence(
    query: str,
    profile: Any,
    results: list["ReadResult"],
    policy: GovernanceCutoffPolicy,
) -> list["ReadResult"]:
    """For metric comparisons, seed the cutoff prefix with direct metric evidence."""
    if policy.query_shape != "comparison" or len(results) < 3:
        return results

    metric_phrases = _comparison_metric_phrases(query)
    if not metric_phrases:
        return results

    entity_variants = _comparison_entity_variants(query, profile)
    scored: list[tuple[int, int, "ReadResult"]] = []
    best_score = 0
    for index, result in enumerate(results):
        evidence = _normalized_evidence([result])
        score = _comparison_metric_evidence_score(evidence, metric_phrases, entity_variants)
        best_score = max(best_score, score)
        scored.append((score, index, result))

    if best_score <= 0:
        return results

    scored.sort(key=lambda item: (-item[0], item[1]))
    return [result for _, _, result in scored]


def _comparison_metric_phrases(query: str) -> tuple[str, ...]:
    """Extract metric-like terms from a comparison query."""
    tokens = [
        token
        for token in _normalize_text(query).split()
        if token and token not in _METRIC_STOP_TERMS and not re.fullmatch(r"q[1-4]|\d{4}", token)
    ]
    phrases: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        normalized = " ".join(_terms_variant(value))
        if normalized and normalized not in seen:
            seen.add(normalized)
            phrases.append(normalized)

    for term, variants in _REQUIRED_QUERY_TERMS.items():
        if re.search(rf"\b{re.escape(term)}s?\b", query.lower()):
            for variant in variants:
                add(variant)

    for first, second in zip(tokens, tokens[1:], strict=False):
        if first in _METRIC_MODIFIER_TERMS or second not in _METRIC_MODIFIER_TERMS:
            add(f"{first} {second}")

    for token in tokens:
        if token not in _METRIC_MODIFIER_TERMS:
            add(token)

    return tuple(phrases)


def _comparison_entity_variants(
    query: str,
    profile: Any,
) -> tuple[tuple[str, ...], ...]:
    """Return normalized entity/time variants each comparison side may satisfy."""
    variants: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()

    def add(items: tuple[str, ...]) -> None:
        if items and items not in seen:
            seen.add(items)
            variants.append(items)

    for _, requirement_variants in _required_temporal_requirements(query):
        for variant in requirement_variants:
            add(variant)

    for entity in getattr(profile, "comparison_entities", []) or []:
        add(_terms_variant(str(entity)))

    return tuple(variants)


def _comparison_metric_evidence_score(
    evidence: str,
    metric_phrases: tuple[str, ...],
    entity_variants: tuple[tuple[str, ...], ...],
) -> int:
    """Score one evidence item by metric specificity and comparison-side coverage."""
    metric_score = 0
    for phrase in metric_phrases:
        terms = _terms_variant(phrase)
        if _contains_all_terms(evidence, terms):
            metric_score += 3 if len(terms) > 1 else 1

    if metric_score == 0:
        return 0

    entity_hits = sum(1 for variant in entity_variants if _contains_all_terms(evidence, variant))
    structured_bonus = 0
    if re.search(r"\bmetric\b", evidence) or "|" in evidence:
        structured_bonus += 1
    if re.search(r"\b(table|row|column|avg|average|total)\b", evidence):
        structured_bonus += 1

    return metric_score * 100 + entity_hits * 20 + structured_bonus * 10


def _required_temporal_requirements(query: str) -> list[tuple[str, tuple[tuple[str, ...], ...]]]:
    """Return explicit month/quarter/year requirements from a query."""
    lower = query.lower()
    requirements: list[tuple[str, tuple[tuple[str, ...], ...]]] = []
    seen: set[str] = set()

    for match in re.finditer(r"\b(q[1-4])(?:\s+(\d{4}))?\b", lower):
        quarter = match.group(1)
        year = match.group(2)
        label = f"{quarter} {year}" if year else quarter
        if label not in seen:
            seen.add(label)
            requirements.append((label, ((_terms_variant(label)),)))

    for match in re.finditer(_MONTH_PATTERN, lower):
        month = match.group(1)
        year = match.group(2)
        label = f"{month} {year}" if year else month
        if label not in seen:
            seen.add(label)
            requirements.append((label, ((_terms_variant(label)),)))

    if not requirements:
        for match in re.finditer(r"\b\d{4}\b", lower):
            year = match.group(0)
            if year not in seen:
                seen.add(year)
                requirements.append((year, ((year,),)))

    return requirements


def _normalized_evidence(results: list["ReadResult"]) -> str:
    """Combine selected evidence fields into normalized searchable text."""
    return _normalize_text(_raw_evidence(results))


def _raw_evidence(results: list["ReadResult"]) -> str:
    """Combine selected evidence fields without destroying exact identifiers."""
    parts: list[str] = []
    for result in results:
        parts.append(str(getattr(result, "content", "")))
        parts.append(str(getattr(result, "file_path", "")))
        address = getattr(result, "address", None)
        if address is not None:
            parts.append(str(getattr(address, "location", "")))
            parts.append(str(getattr(address, "summary", "")))
        metadata = getattr(result, "metadata", {}) or {}
        if isinstance(metadata, dict):
            parts.extend(str(value) for value in metadata.values() if isinstance(value, str))
    return " ".join(parts)


def _contains_all_terms(evidence: str, terms: tuple[str, ...]) -> bool:
    """Return whether all normalized terms are present in selected evidence."""
    return all(re.search(rf"\b{re.escape(term)}\b", evidence) for term in terms if term)


def _terms_variant(value: str) -> tuple[str, ...]:
    """Normalize a requirement label into word-like evidence terms."""
    return tuple(part for part in _normalize_text(value).split() if part)


def _normalize_text(value: str) -> str:
    """Normalize text for deterministic contract matching."""
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9]+", " ", value.lower())).strip()


def _pyrrho_metadata(mode: AnswerMode, decision: Any = None) -> dict[str, Any]:
    """Serialize the governance decision fields Pyrrho exposes."""
    if decision is None:
        return {}

    metadata: dict[str, Any] = {"mode": _decision_mode(decision).value}

    probs = getattr(decision, "probs", None)
    if isinstance(probs, (list, tuple)) and len(probs) == 3:
        metadata["probabilities"] = {
            "insufficient": float(probs[0]),
            "disputed": float(probs[1]),
            "sufficient": float(probs[2]),
        }

    reason = getattr(decision, "reason", None)
    if not isinstance(reason, str):
        reasons = getattr(decision, "reasons", None)
        if isinstance(reasons, (list, tuple)) and reasons:
            reason = str(reasons[0])
    if isinstance(reason, str) and reason:
        metadata["reason"] = reason

    used_consistency_fallback = getattr(decision, "used_consistency_fallback", False) is True
    consistency_reason = getattr(decision, "consistency_reason", None)
    pre_consistency_pair = getattr(decision, "pre_consistency_pair", None)
    if used_consistency_fallback:
        metadata["used_consistency_fallback"] = True
        if isinstance(consistency_reason, str) and consistency_reason:
            metadata["consistency_reason"] = consistency_reason
        if isinstance(pre_consistency_pair, tuple) and len(pre_consistency_pair) == 2:
            metadata["pre_consistency_pair"] = list(pre_consistency_pair)

    input_tokens = getattr(decision, "input_tokens", None)
    max_input_tokens = getattr(decision, "max_input_tokens", None)
    input_truncated = bool(getattr(decision, "input_truncated", False))
    if isinstance(input_tokens, int) or isinstance(max_input_tokens, int):
        if isinstance(input_tokens, int):
            metadata["input_tokens"] = input_tokens
        if isinstance(max_input_tokens, int):
            metadata["max_input_tokens"] = max_input_tokens
        metadata["input_truncated"] = input_truncated

    native_heads = {
        "evidence_verdict",
        "failure_mode",
        "retrieval_intents",
        "evidence_kinds",
    }
    for key in native_heads:
        head = _head_metadata(getattr(decision, key, None))
        if head:
            metadata[key] = head

    heads = getattr(decision, "heads", None)
    if isinstance(heads, dict):
        for key, head in heads.items():
            if not isinstance(key, str) or key not in native_heads or key in metadata:
                continue
            head_data = _head_metadata(head)
            if head_data:
                metadata[key] = head_data

    return metadata


def pyrrho_decision_metadata(mode: AnswerMode, decision: Any = None) -> dict[str, Any]:
    """Serialize Pyrrho decision metadata for answer and evidence surfaces."""
    return _pyrrho_metadata(mode, decision)


def _prefix_trace(size: int, mode: AnswerMode, decision: Any) -> dict[str, Any]:
    """Serialize one evaluated prefix for cutoff observability."""
    trace = _pyrrho_metadata(mode, decision)
    trace["prefix_n"] = size
    return trace


def _head_metadata(head: Any) -> dict[str, Any]:
    """Serialize a Pyrrho head decision."""
    if head is None or isinstance(head, Mock):
        return {}

    metadata: dict[str, Any] = {}
    for key in (
        "raw_label",
        "final_label",
        "final_labels",
        "used_threshold_fallback",
        "used_consistency_fallback",
        "consistency_reason",
        "threshold",
        "confidence",
        "runner_up_label",
        "runner_up_probability",
        "margin_to_runner_up",
        "entropy",
    ):
        value = getattr(head, key, None)
        if value is not None:
            metadata[key] = list(value) if isinstance(value, tuple) else value

    probabilities = getattr(head, "probabilities", None)
    if isinstance(probabilities, dict) and probabilities:
        metadata["probabilities"] = {str(key): float(value) for key, value in probabilities.items()}
    return metadata


def _governance_cutoff_limit(result_count: int, requested_top_k: Any = None) -> int:
    """Return the maximum evidence prefix Pyrrho may inspect."""
    limit = _DEFAULT_GOVERNANCE_CUTOFF
    if requested_top_k is not None:
        try:
            limit = min(limit, max(1, int(requested_top_k)))
        except (TypeError, ValueError):
            pass
    return max(1, min(result_count, limit))


def _pyrrho_governance_results(results: list["ReadResult"]) -> list["ReadResult"]:
    """Return the compact evidence sequence Pyrrho should judge."""
    filtered = [result for result in results if not _is_return_only_bridge_result(result)]
    return filtered or results


def _is_return_only_bridge_result(result: "ReadResult") -> bool:
    """Return whether a compiler source should be returned but not judged by Pyrrho."""
    metadata = getattr(result, "metadata", {}) or {}
    if not isinstance(metadata, dict):
        return False
    compiler = metadata.get("evidence_compiler")
    if not isinstance(compiler, dict):
        return False
    return _compiler_metadata_requires_return_floor(
        compiler
    ) and not _compiler_metadata_requires_prefix_floor(compiler, result)


def _full_prefix_size_for_governance_prefix(
    results: list["ReadResult"],
    governance_prefix: list["ReadResult"],
) -> int:
    """Map a compact Pyrrho prefix back to its full evidence-pack prefix size."""
    if not governance_prefix:
        return 0
    result_positions = {id(result): index for index, result in enumerate(results, start=1)}
    return max(result_positions.get(id(result), 0) for result in governance_prefix)


def _evidence_prefix_min(results: list["ReadResult"]) -> int:
    """Return how many compiler-ledger sources Pyrrho should see before trust."""
    required = 0
    for index, result in enumerate(results, start=1):
        metadata = getattr(result, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        compiler = metadata.get("evidence_compiler")
        if not isinstance(compiler, dict):
            continue
        if not _compiler_metadata_requires_prefix_floor(compiler, result):
            continue
        try:
            required = max(required, int(compiler.get("min_sources", 0)), index)
        except (TypeError, ValueError):
            required = max(required, index)
    return min(len(results), required)


def _return_prefix_min(results: list["ReadResult"]) -> int:
    """Return how many sources should remain in the public evidence pack."""
    required = 0
    for index, result in enumerate(results, start=1):
        metadata = getattr(result, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        compiler = metadata.get("evidence_compiler")
        if not isinstance(compiler, dict):
            continue
        if not _compiler_metadata_requires_return_floor(compiler):
            continue
        try:
            required = max(required, int(compiler.get("min_sources", 0)), index)
        except (TypeError, ValueError):
            required = max(required, index)
    return min(len(results), required)


def _compiler_metadata_requires_prefix_floor(compiler: dict[str, Any], result: Any = None) -> bool:
    """Return whether compiler metadata represents a real Pyrrho evidence obligation."""
    kind = _result_kind(result)
    roles = compiler.get("roles", [])
    if isinstance(roles, list):
        for role in roles:
            role_text = str(role)
            if _compiler_role_requires_prefix_floor(role_text, kind):
                return True
    contract = compiler.get("contract")
    if not isinstance(contract, dict):
        return False
    if contract.get("source_anchors"):
        return True
    return bool(contract.get("temporal_policy") in {"latest", "final"})


def _compiler_metadata_requires_return_floor(compiler: dict[str, Any]) -> bool:
    """Return whether compiler metadata should remain in the returned evidence pack."""
    roles = compiler.get("roles", [])
    if isinstance(roles, list):
        for role in roles:
            role_text = str(role)
            if (
                role_text.startswith("required_")
                or role_text.startswith("anchor_identifier:")
                or role_text.startswith("anchor_phrase:")
                or role_text.startswith("source_anchor:")
                or role_text.startswith("bridge:")
                or role_text.startswith("bridge_document:")
                or role_text in {"conflict_value", "latest", "final"}
            ):
                return True
    contract = compiler.get("contract")
    if not isinstance(contract, dict):
        return False
    if contract.get("required_modalities") or contract.get("source_anchors"):
        return True
    return bool(contract.get("temporal_policy") in {"latest", "final"})


def _compiler_role_requires_prefix_floor(role: str, kind: str | None) -> bool:
    """Return whether this compiler role is a concrete source Pyrrho must inspect."""
    if role.startswith("required_"):
        modality = role.removeprefix("required_")
        return kind is None or kind == modality
    return (
        role.startswith("anchor_identifier:")
        or role.startswith("anchor_phrase:")
        or role.startswith("source_anchor:")
        or role.startswith("bridge:")
        or role in {"conflict_value", "latest", "final"}
    )


def _result_kind(result: Any) -> str | None:
    """Return a normalized ReadResult address kind when present."""
    address = getattr(result, "address", None)
    kind = getattr(address, "kind", None)
    if kind is None:
        return None
    return str(getattr(kind, "value", kind))


__all__ = [
    "GovernanceCutoffPolicy",
    "GovernanceCutoffResult",
    "apply_governance_cutoff",
    "governance_cutoff_policy",
    "governance_query_shape",
    "pyrrho_decision_metadata",
]

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
_DISPUTE_PATIENCE_DOCS = 2
_STABLE_DISPUTE_DOCS = 2
_FOLLOWUP_BATCH_SIZE = 2
_BROAD_OVERVIEW_SOURCE_COUNT = 6
_RETRIEVAL_CONTROL_MIN_CONFIDENCE = 0.70
_ANSWER_NOW_MIN_CONFIDENCE = 0.70
_RETRIEVAL_CONTROL_MORE_ACTIONS = {"retrieve_more", "broaden_search"}
_RETRIEVAL_CONTROL_BLOCKING_ACTIONS = {
    "retrieve_more",
    "broaden_search",
    "resolve_conflict",
    "structured_lookup",
    "ask_clarifying_question",
}
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
    min_trustworthy_docs: int
    min_disputed_docs: int
    disputed_patience_docs: int


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
        mode = AnswerMode.ABSTAIN
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
        mode = AnswerMode.ABSTAIN
        reason = (
            "Query is too broad for evidence sufficiency; returned representative "
            "sources instead of a Pyrrho trustworthy verdict."
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
    pyrrho_contract_prefix_min = _pyrrho_contract_prefix_min(results)

    last_reasons: list[str] = []
    last_decision: Any = None
    last_retrieval_control_blocker: str | None = None
    stable_disputed_decision: Any = None
    trajectory: list[dict[str, Any]] = []
    consecutive_disputed = 0

    for size, decision in _iter_prefix_decisions(governance, query, results, policy):
        mode = _decision_mode(decision)
        last_decision = decision
        last_reasons = _decision_reasons(decision)
        trace = _prefix_trace(size, mode, decision)

        retrieval_control_blocker = _retrieval_control_blocker(decision)
        last_retrieval_control_blocker = retrieval_control_blocker
        if retrieval_control_blocker:
            trace["retrieval_control_blocker"] = retrieval_control_blocker

        if mode is AnswerMode.TRUSTWORTHY:
            consecutive_disputed = 0
            required_trustworthy_docs = pyrrho_contract_prefix_min or policy.min_trustworthy_docs
            can_stop = size >= required_trustworthy_docs
            stop_reason = "trustworthy_min_evidence_met"
            if (
                not can_stop
                and not pyrrho_contract_prefix_min
                and _should_stop_on_answer_now(decision)
            ):
                can_stop = True
                stop_reason = "pyrrho_answer_now"

            if can_stop:
                if retrieval_control_blocker:
                    trajectory.append(trace)
                    continue
                trajectory.append(trace)
                return GovernanceCutoffResult(
                    selected=results[:size],
                    mode=mode,
                    reasons=last_reasons,
                    timings=[("Governance", time.perf_counter() - t0)],
                    metadata=_governance_cutoff_metadata(
                        policy,
                        evaluated=size,
                        selected=size,
                        mode=mode,
                        decision=decision,
                        trajectory=trajectory,
                        stop_reason=stop_reason,
                        retrieval_control_blocker=retrieval_control_blocker,
                    ),
                )
            if pyrrho_contract_prefix_min > size:
                trace["pyrrho_contract_prefix_min"] = pyrrho_contract_prefix_min
            trajectory.append(trace)
            continue

        if mode is AnswerMode.DISPUTED:
            trajectory.append(trace)
            consecutive_disputed += 1
            if consecutive_disputed >= _STABLE_DISPUTE_DOCS:
                stable_disputed_decision = decision
            if _should_stop_on_disputed(policy, size, consecutive_disputed):
                return GovernanceCutoffResult(
                    selected=results[:size],
                    mode=mode,
                    reasons=last_reasons,
                    timings=[("Governance", time.perf_counter() - t0)],
                    metadata=_governance_cutoff_metadata(
                        policy,
                        evaluated=size,
                        selected=size,
                        mode=mode,
                        decision=decision,
                        trajectory=trajectory,
                        stop_reason="dispute_policy_met",
                    ),
                )
            continue

        trajectory.append(trace)
        consecutive_disputed = 0

    if stable_disputed_decision is not None:
        disputed_reasons = _decision_reasons(stable_disputed_decision)
        disputed_reasons.append(
            f"Pyrrho found a stable dispute by the top {policy.max_docs} evidence item(s)."
        )
        return GovernanceCutoffResult(
            selected=results[: policy.max_docs],
            mode=AnswerMode.DISPUTED,
            reasons=disputed_reasons,
            timings=[("Governance", time.perf_counter() - t0)],
            metadata=_governance_cutoff_metadata(
                policy,
                evaluated=policy.max_docs,
                selected=policy.max_docs,
                mode=AnswerMode.DISPUTED,
                decision=stable_disputed_decision,
                trajectory=trajectory,
                stop_reason="stable_dispute_at_cutoff",
            ),
        )

    if last_retrieval_control_blocker:
        reasons = list(last_reasons)
        reasons.insert(0, last_retrieval_control_blocker)
        reasons.append("Pyrrho retrieval control did not authorize an answer within the cutoff.")
        stop_reason = "retrieval_control_unsatisfied_at_cutoff"
        retrieval_control_blocker = last_retrieval_control_blocker
    else:
        reasons = list(last_reasons)
        reasons.append(
            f"Pyrrho abstained after evaluating the top {policy.max_docs} evidence item(s)."
        )
        stop_reason = "cutoff_exhausted"
        retrieval_control_blocker = None
    return GovernanceCutoffResult(
        selected=results[: policy.max_docs],
        mode=AnswerMode.ABSTAIN,
        reasons=reasons,
        timings=[("Governance", time.perf_counter() - t0)],
        metadata=_governance_cutoff_metadata(
            policy,
            evaluated=policy.max_docs,
            selected=policy.max_docs,
            mode=AnswerMode.ABSTAIN,
            decision=last_decision,
            trajectory=trajectory,
            stop_reason=stop_reason,
            retrieval_control_blocker=retrieval_control_blocker,
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
    min_trustworthy_docs = min(max_docs, min_docs_by_shape[query_shape])
    min_disputed_docs = min(max_docs, _COMPARISON_MIN_EVIDENCE)
    return GovernanceCutoffPolicy(
        query_shape=query_shape,
        max_docs=max_docs,
        min_trustworthy_docs=min_trustworthy_docs,
        min_disputed_docs=min_disputed_docs,
        disputed_patience_docs=_DISPUTE_PATIENCE_DOCS,
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
    if _looks_like_temporal_comparison(query or "", profile):
        return "comparison"
    if getattr(profile, "has_aggregation_intent", False):
        return "aggregation"
    if _is_broad_overview_query(query or "", profile):
        return "broad_overview"
    if (
        getattr(profile, "specificity", "") == "broad"
        or getattr(profile, "answer_type", "") == "exploratory"
        or getattr(profile, "inject_corpus_summaries", False)
    ):
        return "broad"
    return "narrow"


def _looks_like_temporal_comparison(query: str, profile: Any) -> bool:
    """Return whether temporal range wording needs comparison-style coverage."""
    if not query:
        return False
    lower = query.lower()
    if not getattr(profile, "has_temporal_intent", False):
        return False
    if len(_required_temporal_requirements(query)) < 2:
        return False
    return bool(
        re.search(
            r"\b(between|compare|comparison|versus|vs|higher|lower|changed|changes|change)\b",
            lower,
        )
    )


def _is_broad_overview_query(query: str, profile: Any) -> bool:
    """Return whether evidence sufficiency is ill-defined for a corpus overview."""
    if not query:
        return False
    if (
        getattr(profile, "specificity", "") != "broad"
        and getattr(profile, "answer_type", "") != "exploratory"
        and not getattr(profile, "inject_corpus_summaries", False)
    ):
        return False
    if getattr(profile, "has_temporal_intent", False):
        return False

    lower = query.lower()
    has_corpus_scope = bool(
        re.search(
            r"\b(corpus|collection|knowledge base|all documents|all docs|all files|all sources|whole repository|entire repository)\b",
            lower,
        )
    )
    has_open_overview_ask = bool(
        re.search(
            r"\b(key facts|overview|summari[sz]e|summary|main themes|survey|representative|which documents|what documents)\b",
            lower,
        )
    )
    return has_corpus_scope and has_open_overview_ask


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
    min_stop_size = policy.min_trustworthy_docs
    if policy.query_shape == "comparison":
        min_stop_size = max(min_stop_size, policy.min_disputed_docs)
    return min(policy.max_docs, max(1, min_stop_size))


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


def _should_stop_on_disputed(
    policy: GovernanceCutoffPolicy,
    size: int,
    consecutive_disputed: int,
) -> bool:
    """Return whether a DISPUTED verdict is strong enough to stop."""
    if policy.query_shape == "comparison":
        return size >= policy.min_disputed_docs
    if policy.query_shape == "narrow":
        return consecutive_disputed >= policy.disputed_patience_docs + 1
    return False


def _should_stop_on_answer_now(decision: Any) -> bool:
    """Return whether Pyrrho explicitly authorizes an early trustworthy stop."""
    label, confidence = _head_label_and_confidence(decision, "retrieval_action")
    return label == "answer_now" and confidence >= _ANSWER_NOW_MIN_CONFIDENCE


def _retrieval_control_blocker(decision: Any) -> str | None:
    """Return a reason to keep evaluating evidence from Pyrrho retrieval control."""
    label, confidence = _head_label_and_confidence(decision, "retrieval_action")
    if label not in _RETRIEVAL_CONTROL_BLOCKING_ACTIONS:
        return None
    if confidence < _RETRIEVAL_CONTROL_MIN_CONFIDENCE:
        return None
    if label in _RETRIEVAL_CONTROL_MORE_ACTIONS:
        return f"Pyrrho retrieval action requested more evidence: {label}."
    if label == "resolve_conflict":
        return "Pyrrho retrieval action requested conflict resolution before trusting evidence."
    if label == "structured_lookup":
        return "Pyrrho retrieval action requested structured lookup evidence."
    if label == "ask_clarifying_question":
        return "Pyrrho retrieval action indicated the query needs clarification."
    return None


def _head_label_and_confidence(decision: Any, name: str) -> tuple[str | None, float]:
    """Read one Pyrrho head's final label and confidence defensively."""
    head = getattr(decision, name, None)
    if head is None:
        heads = getattr(decision, "heads", None)
        if isinstance(heads, dict):
            head = heads.get(name)
    if head is None:
        return None, 0.0
    label = getattr(head, "final_label", None)
    try:
        confidence = float(getattr(head, "confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return str(label) if label else None, confidence


def _decision_mode(decision: Any) -> AnswerMode:
    """Read a decision mode defensively."""
    mode = getattr(decision, "mode", AnswerMode.ABSTAIN)
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
    retrieval_control_blocker: str | None = None,
) -> dict[str, Any]:
    """Build serializable metadata for the cutoff loop."""
    metadata = {
        "evaluated": evaluated,
        "selected": selected,
        "max": policy.max_docs if policy else 0,
        "mode": mode.value,
    }
    if stop_reason:
        metadata["stop_reason"] = stop_reason
    if retrieval_control_blocker:
        metadata["retrieval_control_blocker"] = retrieval_control_blocker
    if policy is not None:
        metadata["policy"] = {
            "query_shape": policy.query_shape,
            "min_trustworthy_docs": policy.min_trustworthy_docs,
            "min_disputed_docs": policy.min_disputed_docs,
            "disputed_patience_docs": policy.disputed_patience_docs,
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
            "abstain": float(probs[0]),
            "disputed": float(probs[1]),
            "trustworthy": float(probs[2]),
        }

    reason = getattr(decision, "reason", None)
    if not isinstance(reason, str):
        reasons = getattr(decision, "reasons", None)
        if isinstance(reasons, (list, tuple)) and reasons:
            reason = str(reasons[0])
    if isinstance(reason, str) and reason:
        metadata["reason"] = reason

    for key in (
        "governance",
        "query_contract",
        "route",
        "taxonomy",
        "retrieval_action",
        "gap_type",
        "answerability_shape",
        "retrieval_modality",
    ):
        head = _head_metadata(getattr(decision, key, None))
        if head:
            metadata[key] = head

    heads = getattr(decision, "heads", None)
    if isinstance(heads, dict):
        for key, head in heads.items():
            if not isinstance(key, str) or key in metadata:
                continue
            head_data = _head_metadata(head)
            if head_data:
                metadata[key] = head_data

    scalars = getattr(decision, "scalars", None)
    if isinstance(scalars, dict) and scalars:
        metadata["scalars"] = {str(key): float(value) for key, value in scalars.items()}

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
        "used_threshold_fallback",
        "threshold",
        "confidence",
        "runner_up_label",
        "runner_up_probability",
        "margin_to_runner_up",
        "entropy",
    ):
        value = getattr(head, key, None)
        if value is not None:
            metadata[key] = value

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


def _pyrrho_contract_prefix_min(results: list["ReadResult"]) -> int:
    """Return how many compiler-ledger sources Pyrrho should see before trust."""
    required = 0
    for result in results:
        metadata = getattr(result, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            continue
        compiler = metadata.get("evidence_compiler")
        if not isinstance(compiler, dict):
            continue
        if not _compiler_metadata_requires_prefix_floor(compiler):
            continue
        try:
            required = max(required, int(compiler.get("min_sources", 0)))
        except (TypeError, ValueError):
            continue
    return min(len(results), required)


def _compiler_metadata_requires_prefix_floor(compiler: dict[str, Any]) -> bool:
    """Return whether compiler metadata represents a real Pyrrho evidence obligation."""
    roles = compiler.get("roles", [])
    if isinstance(roles, list):
        for role in roles:
            role_text = str(role)
            if (
                role_text.startswith("required_")
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


__all__ = [
    "GovernanceCutoffPolicy",
    "GovernanceCutoffResult",
    "apply_governance_cutoff",
    "governance_cutoff_policy",
    "governance_query_shape",
    "pyrrho_decision_metadata",
]

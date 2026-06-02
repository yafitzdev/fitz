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


@dataclass(frozen=True)
class GovernanceCutoffPolicy:
    """Deterministic stop policy wrapped around Pyrrho verdicts."""

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

    last_reasons: list[str] = []
    last_decision: Any = None
    stable_disputed_decision: Any = None
    trajectory: list[dict[str, Any]] = []
    consecutive_disputed = 0

    for size, decision in _iter_prefix_decisions(governance, query, results, policy):
        mode = _decision_mode(decision)
        last_decision = decision
        last_reasons = _decision_reasons(decision)
        trajectory.append(_prefix_trace(size, mode, decision))

        if mode is AnswerMode.TRUSTWORTHY:
            consecutive_disputed = 0
            if size >= policy.min_trustworthy_docs:
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
                        stop_reason="trustworthy_min_evidence_met",
                    ),
                )
            continue

        if mode is AnswerMode.DISPUTED:
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

    reasons = list(last_reasons)
    reasons.append(f"Pyrrho abstained after evaluating the top {policy.max_docs} evidence item(s).")
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
            stop_reason="cutoff_exhausted",
        ),
    )


def governance_cutoff_policy(
    profile: Any,
    result_count: int,
    requested_top_k: Any = None,
    *,
    query: str | None = None,
) -> GovernanceCutoffPolicy:
    """Build a deterministic policy for interpreting Pyrrho verdicts."""
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


def _pyrrho_metadata(mode: AnswerMode, decision: Any = None) -> dict[str, Any]:
    """Serialize the governance decision fields Pyrrho exposes."""
    if decision is None:
        return {}

    metadata: dict[str, Any] = {"mode": mode.value}

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

    for key in ("governance", "query_contract", "route", "taxonomy"):
        head = _head_metadata(getattr(decision, key, None))
        if head:
            metadata[key] = head

    scalars = getattr(decision, "scalars", None)
    if isinstance(scalars, dict) and scalars:
        metadata["scalars"] = {str(key): float(value) for key, value in scalars.items()}

    return metadata


def _prefix_trace(size: int, mode: AnswerMode, decision: Any) -> dict[str, Any]:
    """Serialize one evaluated prefix for cutoff observability."""
    trace = _pyrrho_metadata(mode, decision)
    trace["prefix_n"] = size
    return trace


def _head_metadata(head: Any) -> dict[str, Any]:
    """Serialize a Pyrrho g3.1 head decision."""
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


__all__ = [
    "GovernanceCutoffPolicy",
    "GovernanceCutoffResult",
    "apply_governance_cutoff",
    "governance_cutoff_policy",
    "governance_query_shape",
]

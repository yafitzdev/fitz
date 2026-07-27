# benchmarks/fitz_bench/validators.py
"""Evidence-first validators for the retrieval benchmark."""

from __future__ import annotations

import re
from typing import Any

from benchmarks.fitz_bench.models import (
    BenchmarkCase,
    CaseMetrics,
    EvidenceExpectation,
    ValidationResult,
    normalize_mode,
)


def validate_case(
    case: BenchmarkCase,
    evidence_pack: dict[str, Any],
    *,
    ranked_items: list[dict[str, Any]] | None = None,
    signals: dict[str, Any] | None = None,
) -> ValidationResult:
    """Validate compiled retrieval, fixed delivery, Pyrrho, and query signals."""
    retrieval_failures: list[str] = []
    delivery_failures: list[str] = []
    governance_failures: list[str] = []
    query_shape_failures: list[str] = []
    delivered_items = list(evidence_pack.get("items") or [])
    retrieval_items = list(ranked_items) if ranked_items is not None else delivered_items
    signal_root = signals if signals is not None else evidence_pack
    mode = normalize_mode(evidence_pack.get("mode"))

    mode_match: bool | None = None
    if case.expected_mode is not None:
        mode_match = mode == case.expected_mode
        if not mode_match:
            governance_failures.append(f"Expected mode {case.expected_mode!r}, got {mode!r}.")

    matched_required: list[int | None] = []
    matched_delivery_required: list[int | None] = []
    for expectation in case.required_evidence:
        rank = _first_matching_rank(expectation, retrieval_items)
        matched_required.append(rank)
        if rank is None:
            retrieval_failures.append(
                f"Missing required evidence: {_expectation_label(expectation)}."
            )
        delivery_rank = _first_matching_rank(expectation, delivered_items)
        matched_delivery_required.append(delivery_rank)
        if delivery_rank is None:
            delivery_failures.append(
                f"Missing delivered evidence: {_expectation_label(expectation)}."
            )

    matched_forbidden: list[int] = []
    matched_delivery_forbidden: list[int] = []
    for expectation in case.forbidden_evidence:
        rank = _first_matching_forbidden_rank(
            expectation,
            retrieval_items,
            case.required_evidence,
        )
        if rank is not None:
            matched_forbidden.append(rank)
            retrieval_failures.append(
                f"Forbidden evidence matched at rank {rank}: {_expectation_label(expectation)}."
            )
        delivery_rank = _first_matching_forbidden_rank(
            expectation,
            delivered_items,
            case.required_evidence,
        )
        if delivery_rank is not None:
            matched_delivery_forbidden.append(delivery_rank)
            delivery_failures.append(
                "Forbidden delivered evidence matched at rank "
                f"{delivery_rank}: {_expectation_label(expectation)}."
            )

    for path, expected in case.expected_signals.items():
        actual = _get_path(signal_root, path)
        if actual != expected:
            query_shape_failures.append(f"Expected signal {path}={expected!r}, got {actual!r}.")

    failures = delivery_failures + governance_failures + query_shape_failures
    metrics = _metrics(
        case,
        matched_required,
        matched_forbidden,
        mode_match,
        retrieval_passed=not retrieval_failures,
        delivery_passed=not delivery_failures,
        query_shape_passed=not query_shape_failures,
    )
    return ValidationResult(
        passed=not failures,
        failures=tuple(failures),
        metrics=metrics,
        matched_required=tuple(matched_required),
        matched_forbidden=tuple(matched_forbidden),
        matched_delivery_required=tuple(matched_delivery_required),
        matched_delivery_forbidden=tuple(matched_delivery_forbidden),
        retrieval_failures=tuple(retrieval_failures),
        delivery_failures=tuple(delivery_failures),
        governance_failures=tuple(governance_failures),
        query_shape_failures=tuple(query_shape_failures),
    )


def _metrics(
    case: BenchmarkCase,
    matched_required: list[int | None],
    matched_forbidden: list[int],
    mode_match: bool | None,
    *,
    retrieval_passed: bool,
    delivery_passed: bool,
    query_shape_passed: bool,
) -> CaseMetrics:
    """Compute deterministic retrieval metrics for one case."""
    ranks = [rank for rank in matched_required if rank is not None]
    first_rank = min(ranks) if ranks else None
    required_total = len(case.required_evidence)
    required_recall = len(ranks) / required_total if required_total else 1.0
    retrieval_evaluated = bool(case.required_evidence or case.forbidden_evidence)
    query_shape_evaluated = bool(case.expected_signals)
    return CaseMetrics(
        retrieval_evaluated=retrieval_evaluated,
        delivery_evaluated=retrieval_evaluated,
        query_shape_evaluated=query_shape_evaluated,
        capability_evaluated=retrieval_evaluated or query_shape_evaluated,
        hit_at_1=first_rank is not None and first_rank <= 1,
        hit_at_3=first_rank is not None and first_rank <= 3,
        hit_at_5=first_rank is not None and first_rank <= 5,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        required_recall=required_recall,
        forbidden_count=len(matched_forbidden),
        mode_match=mode_match,
        retrieval_passed=retrieval_passed,
        delivery_passed=delivery_passed,
        query_shape_passed=query_shape_passed,
        capability_passed=retrieval_passed and query_shape_passed,
    )


def _first_matching_rank(
    expectation: EvidenceExpectation, items: list[dict[str, Any]]
) -> int | None:
    """Return the first evidence rank that satisfies an expectation."""
    for index, item in enumerate(items, start=1):
        if _matches(expectation, item):
            return int(item.get("rank") or index)
    return None


def _first_matching_forbidden_rank(
    expectation: EvidenceExpectation,
    items: list[dict[str, Any]],
    required: tuple[EvidenceExpectation, ...],
) -> int | None:
    """Return first forbidden-only rank, ignoring coarse sections that satisfy required evidence.

    Some benchmark corpora keep old and current dated entries in one Markdown
    section. A returned section should not fail merely because it contains the
    older paragraph when it also contains the required current/final paragraph.
    """
    for index, item in enumerate(items, start=1):
        if not _matches(expectation, item):
            continue
        if any(_matches(required_expectation, item) for required_expectation in required):
            continue
        return int(item.get("rank") or index)
    return None


def _matches(expectation: EvidenceExpectation, item: dict[str, Any]) -> bool:
    """Return whether one evidence item matches one expectation."""
    if expectation.file:
        normalized_file = expectation.file.replace("\\", "/").lower()
        file_path = str(item.get("file_path", "")).replace("\\", "/").lower()
        source_id = str(item.get("source_id", "")).replace("\\", "/").lower()
        if normalized_file not in file_path and normalized_file not in source_id:
            return False

    if expectation.kind and item.get("address_kind") != expectation.kind:
        return False

    if expectation.location_contains:
        needle = _normalize_text(expectation.location_contains)
        location = _normalize_text(str(item.get("address_location", "")))
        if needle not in location:
            return False

    searchable = _normalize_text(
        "\n".join(str(item.get(field, "")) for field in ("excerpt", "content", "address_location"))
    )
    if not all(_normalize_text(fragment) in searchable for fragment in expectation.contains):
        return False
    if expectation.contains_any and not any(
        _normalize_text(fragment) in searchable for fragment in expectation.contains_any
    ):
        return False
    return True


def _normalize_text(value: str) -> str:
    """Collapse benchmark text so YAML phrases can match wrapped Markdown."""
    return re.sub(r"\s+", " ", str(value).lower()).strip()


def _expectation_label(expectation: EvidenceExpectation) -> str:
    """Render an expectation for failure output."""
    parts = []
    if expectation.file:
        parts.append(f"file={expectation.file}")
    if expectation.kind:
        parts.append(f"kind={expectation.kind}")
    if expectation.location_contains:
        parts.append(f"location_contains={expectation.location_contains}")
    if expectation.contains:
        parts.append(f"contains={list(expectation.contains)}")
    if expectation.contains_any:
        parts.append(f"contains_any={list(expectation.contains_any)}")
    return ", ".join(parts) if parts else "<empty expectation>"


def _get_path(root: dict[str, Any], path: str) -> Any:
    """Resolve dotted paths in pack JSON, returning None when absent."""
    current: Any = root
    for part in path.split("."):
        if isinstance(current, dict) and part in current:
            current = current[part]
            continue
        return None
    return current


__all__ = ["validate_case"]

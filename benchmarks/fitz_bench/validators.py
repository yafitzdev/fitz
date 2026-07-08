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


def validate_case(case: BenchmarkCase, evidence_pack: dict[str, Any]) -> ValidationResult:
    """Validate one evidence pack against one benchmark case."""
    failures: list[str] = []
    items = list(evidence_pack.get("items") or [])
    mode = normalize_mode(evidence_pack.get("mode"))

    mode_match: bool | None = None
    if case.expected_mode is not None:
        mode_match = mode == case.expected_mode
        if not mode_match:
            failures.append(f"Expected mode {case.expected_mode!r}, got {mode!r}.")

    matched_required: list[int | None] = []
    for expectation in case.required_evidence:
        rank = _first_matching_rank(expectation, items)
        matched_required.append(rank)
        if rank is None:
            failures.append(f"Missing required evidence: {_expectation_label(expectation)}.")

    matched_forbidden: list[int] = []
    for expectation in case.forbidden_evidence:
        rank = _first_matching_forbidden_rank(
            expectation,
            items,
            case.required_evidence,
        )
        if rank is not None:
            matched_forbidden.append(rank)
            failures.append(
                f"Forbidden evidence matched at rank {rank}: {_expectation_label(expectation)}."
            )

    for path, expected in case.expected_signals.items():
        actual = _get_path(evidence_pack, path)
        if actual != expected:
            failures.append(f"Expected signal {path}={expected!r}, got {actual!r}.")

    metrics = _metrics(case, matched_required, matched_forbidden, mode_match)
    return ValidationResult(
        passed=not failures,
        failures=tuple(failures),
        metrics=metrics,
        matched_required=tuple(matched_required),
        matched_forbidden=tuple(matched_forbidden),
    )


def _metrics(
    case: BenchmarkCase,
    matched_required: list[int | None],
    matched_forbidden: list[int],
    mode_match: bool | None,
) -> CaseMetrics:
    """Compute deterministic retrieval metrics for one case."""
    ranks = [rank for rank in matched_required if rank is not None]
    first_rank = min(ranks) if ranks else None
    required_total = len(case.required_evidence)
    required_recall = len(ranks) / required_total if required_total else 1.0
    return CaseMetrics(
        hit_at_1=first_rank is not None and first_rank <= 1,
        hit_at_3=first_rank is not None and first_rank <= 3,
        hit_at_5=first_rank is not None and first_rank <= 5,
        mrr=(1.0 / first_rank) if first_rank else 0.0,
        required_recall=required_recall,
        forbidden_count=len(matched_forbidden),
        mode_match=mode_match,
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
    return all(_normalize_text(fragment) in searchable for fragment in expectation.contains)


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

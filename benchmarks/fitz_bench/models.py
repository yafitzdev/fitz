# benchmarks/fitz_bench/models.py
"""Data contracts for the fitz-sage retrieval benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

_VALID_MODES = {"sufficient", "disputed", "insufficient"}


def normalize_mode(value: Any) -> str | None:
    """Normalize v2 governance label names used by benchmark cases and packs."""
    if value is None:
        return None
    raw = getattr(value, "value", value)
    normalized = str(raw).strip().lower()
    return normalized if normalized in _VALID_MODES else normalized


@dataclass(frozen=True)
class EvidenceExpectation:
    """One expected or forbidden evidence pattern."""

    file: str | None = None
    kind: str | None = None
    location_contains: str | None = None
    contains: tuple[str, ...] = ()
    contains_any: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "EvidenceExpectation":
        """Build an expectation from YAML data."""
        contains = raw.get("contains") or ()
        if isinstance(contains, str):
            contains = (contains,)
        contains_any = raw.get("contains_any") or ()
        if isinstance(contains_any, str):
            contains_any = (contains_any,)
        return cls(
            file=raw.get("file"),
            kind=raw.get("kind"),
            location_contains=raw.get("location_contains"),
            contains=tuple(str(item) for item in contains),
            contains_any=tuple(str(item) for item in contains_any),
        )


@dataclass(frozen=True)
class BenchmarkCase:
    """One retrieval benchmark case."""

    case_id: str
    domain: str
    query: str
    expected_mode: str | None = None
    required_evidence: tuple[EvidenceExpectation, ...] = ()
    forbidden_evidence: tuple[EvidenceExpectation, ...] = ()
    expected_signals: dict[str, Any] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchmarkCase":
        """Build a benchmark case from YAML data."""
        expected = raw.get("expected") or {}
        tags = raw.get("tags") or ()
        return cls(
            case_id=str(raw["id"]),
            domain=str(raw["domain"]),
            query=str(raw["query"]),
            expected_mode=normalize_mode(expected.get("mode")),
            required_evidence=tuple(
                EvidenceExpectation.from_dict(item)
                for item in expected.get("required_evidence", [])
            ),
            forbidden_evidence=tuple(
                EvidenceExpectation.from_dict(item)
                for item in expected.get("forbidden_evidence", [])
            ),
            expected_signals=dict(raw.get("expected_signals") or {}),
            tags=tuple(str(tag) for tag in tags),
        )


@dataclass(frozen=True)
class CaseMetrics:
    """Deterministic metrics for one case."""

    retrieval_evaluated: bool
    delivery_evaluated: bool
    query_shape_evaluated: bool
    capability_evaluated: bool
    hit_at_1: bool
    hit_at_3: bool
    hit_at_5: bool
    mrr: float
    required_recall: float
    forbidden_count: int
    mode_match: bool | None
    retrieval_passed: bool
    delivery_passed: bool
    query_shape_passed: bool
    capability_passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize metrics."""
        return {
            "retrieval_evaluated": self.retrieval_evaluated,
            "delivery_evaluated": self.delivery_evaluated,
            "query_shape_evaluated": self.query_shape_evaluated,
            "capability_evaluated": self.capability_evaluated,
            "hit_at_1": self.hit_at_1,
            "hit_at_3": self.hit_at_3,
            "hit_at_5": self.hit_at_5,
            "mrr": self.mrr,
            "required_recall": self.required_recall,
            "forbidden_count": self.forbidden_count,
            "mode_match": self.mode_match,
            "retrieval_passed": self.retrieval_passed,
            "delivery_passed": self.delivery_passed,
            "query_shape_passed": self.query_shape_passed,
            "capability_passed": self.capability_passed,
        }


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for one benchmark case."""

    passed: bool
    failures: tuple[str, ...]
    metrics: CaseMetrics
    matched_required: tuple[int | None, ...]
    matched_forbidden: tuple[int, ...]
    matched_delivery_required: tuple[int | None, ...] = ()
    matched_delivery_forbidden: tuple[int, ...] = ()
    retrieval_failures: tuple[str, ...] = ()
    delivery_failures: tuple[str, ...] = ()
    governance_failures: tuple[str, ...] = ()
    query_shape_failures: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Serialize validation result."""
        return {
            "passed": self.passed,
            "failures": list(self.failures),
            "metrics": self.metrics.to_dict(),
            "matched_required": list(self.matched_required),
            "matched_forbidden": list(self.matched_forbidden),
            "matched_delivery_required": list(self.matched_delivery_required),
            "matched_delivery_forbidden": list(self.matched_delivery_forbidden),
            "retrieval_failures": list(self.retrieval_failures),
            "delivery_failures": list(self.delivery_failures),
            "governance_failures": list(self.governance_failures),
            "query_shape_failures": list(self.query_shape_failures),
        }


__all__ = [
    "BenchmarkCase",
    "CaseMetrics",
    "EvidenceExpectation",
    "ValidationResult",
    "normalize_mode",
]

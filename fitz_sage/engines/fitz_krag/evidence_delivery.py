"""Progressively deliver ranked evidence to Pyrrho."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.integrations.pyrrho import (
    OnnxPyrrho,
    answer_mode_from_pyrrho,
    decide,
    decision_payload,
)

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.types import ReadResult
    from fitz_sage.llm.providers.pyrrho_types import GovernanceDecision

_INITIAL_PREFIX_SIZE = 3
_PREFIX_INCREMENT = 2
_TERMINAL_MODES = frozenset({AnswerMode.SUFFICIENT, AnswerMode.DISPUTED})


@dataclass(frozen=True)
class EvidencePrefixEvaluation:
    """One exact Pyrrho decision over a ranked evidence prefix."""

    evidence_count: int
    decision: GovernanceDecision

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_count": self.evidence_count,
            "decision": decision_payload(self.decision),
        }


@dataclass(frozen=True)
class ProgressiveEvidenceDelivery:
    """The terminal or exhausted ranked evidence prefix."""

    selected: tuple[ReadResult, ...]
    decision: GovernanceDecision
    evaluations: tuple[EvidencePrefixEvaluation, ...]

    def metadata(self, *, available: int, limit: int) -> dict[str, Any]:
        """Describe the mechanical delivery loop and preserve every model output."""
        return {
            "available": available,
            "selected": len(self.selected),
            "limit": limit,
            "initial_prefix_size": _INITIAL_PREFIX_SIZE,
            "prefix_increment": _PREFIX_INCREMENT,
            "evaluated_prefixes": [item.evidence_count for item in self.evaluations],
            "trajectory": [item.to_dict() for item in self.evaluations],
        }


def deliver_progressively(
    runtime: OnnxPyrrho,
    query: str,
    ranked_results: Sequence[ReadResult],
) -> ProgressiveEvidenceDelivery:
    """Grow the ranked prefix until Pyrrho returns a terminal verdict."""
    candidates = tuple(ranked_results)
    evaluations: list[EvidencePrefixEvaluation] = []

    for evidence_count in _prefix_sizes(len(candidates)):
        decision = decide(runtime, query, candidates[:evidence_count])
        evaluation = EvidencePrefixEvaluation(
            evidence_count=evidence_count,
            decision=decision,
        )
        evaluations.append(evaluation)
        mode = answer_mode_from_pyrrho(decision)
        if mode in _TERMINAL_MODES or evidence_count == len(candidates):
            return ProgressiveEvidenceDelivery(
                selected=candidates[:evidence_count],
                decision=decision,
                evaluations=tuple(evaluations),
            )

    raise RuntimeError("Progressive evidence delivery produced no Pyrrho decision.")


def _prefix_sizes(result_count: int) -> tuple[int, ...]:
    """Return the 3, +2 schedule, including a shorter final prefix when needed."""
    if result_count <= 0:
        return (0,)

    first = min(_INITIAL_PREFIX_SIZE, result_count)
    sizes = list(range(first, result_count + 1, _PREFIX_INCREMENT))
    if sizes[-1] != result_count:
        sizes.append(result_count)
    return tuple(sizes)


__all__ = [
    "EvidencePrefixEvaluation",
    "ProgressiveEvidenceDelivery",
    "deliver_progressively",
]

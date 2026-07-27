"""Thin integration with Pyrrho's authoritative governance runtime."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pyrrho import GovernanceDecision, Pyrrho, QueryPlan

from fitz_sage.core.answer_mode import AnswerMode

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.types import ReadResult

_VERDICT_TO_MODE = {
    "INSUFFICIENT": AnswerMode.INSUFFICIENT,
    "DISPUTED": AnswerMode.DISPUTED,
    "SUFFICIENT": AnswerMode.SUFFICIENT,
}


def create_pyrrho(spec: str) -> Pyrrho:
    """Build Pyrrho from Fitz-Sage's provider/model configuration syntax."""
    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Governance must be 'pyrrho' or 'pyrrho/<package>'.")
    provider, separator, model_spec = spec.strip().partition("/")
    if provider != "pyrrho":
        raise ValueError(
            f"Unknown governance provider: {provider!r}. "
            "Supported: 'pyrrho' or 'pyrrho/<package>'."
        )
    if separator and not model_spec.strip():
        raise ValueError("Pyrrho package specification cannot be empty.")
    return Pyrrho(model_spec.strip()) if separator else Pyrrho()


def pyrrho_evidence(results: Sequence["ReadResult"]) -> list[dict[str, str]]:
    """Expose retrieved evidence without rewriting or normalizing its content."""
    evidence: list[dict[str, str]] = []
    for result in results:
        address = result.address
        evidence.append(
            {
                "source_id": str(address.source_id),
                "text": str(result.content),
            }
        )
    return evidence


def decide(
    runtime: Pyrrho,
    query: str,
    results: Sequence["ReadResult"],
) -> GovernanceDecision:
    """Ask Pyrrho for the authoritative verdict over exactly these results."""
    decision = runtime.decide(query, pyrrho_evidence(results))
    answer_mode_from_pyrrho(decision)
    return decision


def answer_mode_from_pyrrho(decision: GovernanceDecision) -> AnswerMode:
    """Mechanically map Pyrrho's verdict into Fitz-Sage's output type."""
    try:
        return _VERDICT_TO_MODE[decision.verdict]
    except (AttributeError, KeyError) as exc:
        verdict = getattr(decision, "verdict", None)
        raise ValueError(f"Pyrrho returned an unknown verdict: {verdict!r}.") from exc


def decision_payload(decision: GovernanceDecision) -> dict[str, Any]:
    """Return Pyrrho's exact public serialized decision."""
    payload = decision.to_dict()
    if not isinstance(payload, dict):
        raise TypeError("Pyrrho decision.to_dict() must return a dictionary.")
    return payload


__all__ = [
    "GovernanceDecision",
    "Pyrrho",
    "QueryPlan",
    "answer_mode_from_pyrrho",
    "create_pyrrho",
    "decide",
    "decision_payload",
    "pyrrho_evidence",
]

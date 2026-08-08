"""Serializable outputs produced by the managed Pyrrho model."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class HeadDecision:
    """One mutually exclusive Pyrrho head."""

    raw_label: str
    final_label: str
    probabilities: dict[str, float]
    confidence: float
    runner_up_label: str
    runner_up_probability: float
    margin_to_runner_up: float
    entropy: float
    threshold: float | None = None
    threshold_applied: bool = False
    consistency_applied: bool = False
    consistency_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_label": self.raw_label,
            "final_label": self.final_label,
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "runner_up_label": self.runner_up_label,
            "runner_up_probability": self.runner_up_probability,
            "margin_to_runner_up": self.margin_to_runner_up,
            "entropy": self.entropy,
            "threshold": self.threshold,
            "threshold_applied": self.threshold_applied,
            "consistency_applied": self.consistency_applied,
            "consistency_reason": self.consistency_reason,
        }


@dataclass(frozen=True)
class MultiLabelDecision:
    """One independent-label Pyrrho head."""

    raw_label: str
    final_label: str
    final_labels: tuple[str, ...]
    probabilities: dict[str, float]
    confidence: float
    runner_up_label: str
    runner_up_probability: float
    margin_to_runner_up: float
    entropy: float
    threshold: float
    threshold_applied: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_label": self.raw_label,
            "final_label": self.final_label,
            "final_labels": list(self.final_labels),
            "probabilities": dict(self.probabilities),
            "confidence": self.confidence,
            "runner_up_label": self.runner_up_label,
            "runner_up_probability": self.runner_up_probability,
            "margin_to_runner_up": self.margin_to_runner_up,
            "entropy": self.entropy,
            "threshold": self.threshold,
            "threshold_applied": self.threshold_applied,
        }


@dataclass(frozen=True)
class PyrrhoQueryPlan:
    """Pyrrho's query-only native planning heads."""

    retrieval_intents: MultiLabelDecision
    evidence_kinds: MultiLabelDecision
    input_tokens: int
    input_truncated: bool
    max_input_tokens: int

    @property
    def heads(self) -> dict[str, MultiLabelDecision]:
        return {
            "retrieval_intents": self.retrieval_intents,
            "evidence_kinds": self.evidence_kinds,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "retrieval_intents": self.retrieval_intents.to_dict(),
            "evidence_kinds": self.evidence_kinds.to_dict(),
            "input_tokens": self.input_tokens,
            "input_truncated": self.input_truncated,
            "max_input_tokens": self.max_input_tokens,
        }


@dataclass(frozen=True)
class GovernanceDecision:
    """Pyrrho's verdict over one complete evidence set."""

    verdict: str
    reason: str
    probabilities: dict[str, float]
    evidence_verdict: HeadDecision
    failure_mode: HeadDecision
    retrieval_intents: MultiLabelDecision | None = None
    evidence_kinds: MultiLabelDecision | None = None
    input_tokens: int = 0
    input_truncated: bool = False
    max_input_tokens: int = 0
    deterministic: bool = False
    consistency_applied: bool = False
    consistency_reason: str | None = None
    pre_consistency_pair: tuple[str, str] | None = None
    model: dict[str, Any] = field(default_factory=dict)

    @property
    def reasons(self) -> tuple[str, ...]:
        return (self.reason,) if self.reason else ()

    @property
    def heads(self) -> dict[str, HeadDecision | MultiLabelDecision]:
        heads: dict[str, HeadDecision | MultiLabelDecision] = {
            "evidence_verdict": self.evidence_verdict,
            "failure_mode": self.failure_mode,
        }
        if self.retrieval_intents is not None:
            heads["retrieval_intents"] = self.retrieval_intents
        if self.evidence_kinds is not None:
            heads["evidence_kinds"] = self.evidence_kinds
        return heads

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "verdict": self.verdict,
            "reason": self.reason,
            "probabilities": dict(self.probabilities),
            "heads": {name: head.to_dict() for name, head in self.heads.items()},
            "input": {
                "tokens": self.input_tokens,
                "truncated": self.input_truncated,
                "max_tokens": self.max_input_tokens,
            },
            "deterministic": self.deterministic,
            "consistency": {
                "applied": self.consistency_applied,
                "reason": self.consistency_reason,
                "original_pair": (
                    list(self.pre_consistency_pair)
                    if self.pre_consistency_pair is not None
                    else None
                ),
            },
            "model": dict(self.model),
        }


@dataclass(frozen=True)
class PyrrhoModelIdentity:
    """Identity of the exact Pyrrho model artifact selected by Fitz-Sage."""

    model_spec: str
    model_directory: str
    graph: str
    graph_sha256: str
    max_input_tokens: int
    sufficient_threshold: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_spec": self.model_spec,
            "model_directory": self.model_directory,
            "graph": self.graph,
            "graph_sha256": self.graph_sha256,
            "max_input_tokens": self.max_input_tokens,
            "sufficient_threshold": self.sufficient_threshold,
        }


__all__ = [
    "GovernanceDecision",
    "HeadDecision",
    "MultiLabelDecision",
    "PyrrhoModelIdentity",
    "PyrrhoQueryPlan",
]

# tests/unit/test_governance_cutoff.py
"""Unit tests for Pyrrho cutoff policy metadata."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.governance_cutoff import apply_governance_cutoff
from fitz_sage.governance.pyrrho import GovernanceDecision, HeadDecision


def test_cutoff_serializes_pyrrho_g31_metadata():
    """g3.1 head and scalar outputs should survive into evidence-pack metadata."""
    decision = GovernanceDecision(
        mode=AnswerMode.TRUSTWORTHY,
        probs=(0.05, 0.10, 0.85),
        reason="Pyrrho: sources support a confident answer (P=0.85).",
        governance=_head(
            raw_label="TRUSTWORTHY",
            final_label="TRUSTWORTHY",
            probabilities={"ABSTAIN": 0.05, "DISPUTED": 0.10, "TRUSTWORTHY": 0.85},
        ),
        query_contract=_head(
            raw_label="structured_lookup",
            final_label="structured_lookup",
            probabilities={"structured_lookup": 0.88, "representative_overview": 0.12},
        ),
        route=_head(
            raw_label="business_ops",
            final_label="business_ops",
            probabilities={"business_ops": 0.80, "technical_docs": 0.20},
        ),
        taxonomy=_head(
            raw_label="evidence_direct",
            final_label="evidence_direct",
            probabilities={"evidence_direct": 0.72, "evidence_indirect": 0.28},
        ),
        scalars={"retrieval_retry_value": 0.17, "false_trustworthy_risk": 0.09},
    )

    class Governance:
        def decide(self, query: str, contexts: list[SimpleNamespace]) -> GovernanceDecision:
            return decision

    result = apply_governance_cutoff(
        "Which invoice failed?",
        [SimpleNamespace(content="Invoice 17 failed retry validation.")],
        Governance(),
    )

    metadata = result.metadata
    assert metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert metadata["trajectory"][0]["prefix_n"] == 1
    pyrrho = metadata["pyrrho"]
    assert pyrrho["governance"]["raw_label"] == "TRUSTWORTHY"
    assert pyrrho["query_contract"]["final_label"] == "structured_lookup"
    assert pyrrho["route"]["final_label"] == "business_ops"
    assert pyrrho["taxonomy"]["final_label"] == "evidence_direct"
    assert pyrrho["scalars"]["retrieval_retry_value"] == 0.17
    assert pyrrho["scalars"]["false_trustworthy_risk"] == 0.09


def _head(
    *,
    raw_label: str,
    final_label: str,
    probabilities: dict[str, float],
) -> HeadDecision:
    """Build a compact head fixture."""
    sorted_probs = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
    runner_up = sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
    confidence = probabilities[final_label]
    return HeadDecision(
        raw_label=raw_label,
        final_label=final_label,
        used_threshold_fallback=False,
        threshold=None,
        confidence=confidence,
        probabilities=probabilities,
        runner_up_label=runner_up[0],
        runner_up_probability=runner_up[1],
        margin_to_runner_up=confidence - runner_up[1],
        entropy=0.0,
    )

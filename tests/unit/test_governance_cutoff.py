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


def test_contract_gate_waits_for_all_comparison_entities():
    """A comparison verdict cannot stop while one explicit side is missing."""
    profile = SimpleNamespace(
        has_temporal_intent=True,
        has_comparison_intent=True,
        comparison_entities=["Q1", "Q2 2024"],
        answer_type="comparative",
    )
    results = [
        _result("Q1 revenue increased.", "quarterly_summary_q1_2024.md"),
        _result("Q1 customer feedback improved.", "feedback_q1_2024.md"),
        _result("Q2 revenue increased faster.", "quarterly_summary_q2_2024.md"),
    ]

    result = apply_governance_cutoff(
        "What changed between Q1 and Q2 2024?",
        results,
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert len(result.selected) == 3
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert "contract_blocker" in result.metadata["trajectory"][1]


def test_contract_gate_abstains_when_temporal_scope_is_missing():
    """Pyrrho confidence should not certify evidence from the wrong month."""
    profile = SimpleNamespace(has_temporal_intent=True, has_comparison_intent=False)

    result = apply_governance_cutoff(
        "What changed in March 2024?",
        [_result("April feedback improved after the beta release.", "feedback_april_2024.md")],
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] == "contract_unsatisfied_at_cutoff"
    assert "march 2024" in result.reasons[0]


def test_contract_gate_abstains_when_required_metric_is_missing():
    """Metric questions need evidence containing the metric, not just the period."""
    profile = SimpleNamespace(has_temporal_intent=True, has_comparison_intent=False)

    result = apply_governance_cutoff(
        "What was Q4 2024 revenue?",
        [_result("Q4 2024 roadmap targets were postponed.", "roadmap_q4_2024.md")],
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["contract_blocker"]
    assert "revenue" in result.reasons[0]


def test_metric_comparison_cutoff_prefers_exact_table_evidence():
    """Metric comparisons should not stop on weaker prose before exact table rows."""
    profile = SimpleNamespace(
        has_temporal_intent=True,
        has_comparison_intent=True,
        comparison_entities=["Q1", "Q2"],
        answer_type="comparative",
    )
    q2_metrics = _result(
        "| Metric | April | May | Q2 Avg | vs Q1 |\n"
        "| Total Responses | 278 | 312 | 295 | +51% |",
        "quarterly_summary_q2_2024.md",
    )
    q1_conclusion = _result(
        "Q1 established a strong foundation. Customer engagement is increasing "
        "(50% more responses).",
        "quarterly_summary_q1_2024.md",
    )
    q1_metrics = _result(
        "| Metric | January | February | March | Change |\n"
        "| Total Responses | 156 | 189 | 234 | +50% |",
        "quarterly_summary_q1_2024.md",
    )

    result = apply_governance_cutoff(
        "Which quarterly summary had higher total responses, Q1 or Q2?",
        [q2_metrics, q1_conclusion, q1_metrics],
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [q2_metrics, q1_metrics]
    assert q1_conclusion not in result.selected


def test_structured_lookup_exact_identifier_satisfies_retrieval_contract():
    """Source-finding queries should trust exact identifier matches after Pyrrho abstains."""
    profile = SimpleNamespace(query_contract="structured_lookup")
    exact_match = _result(
        "All regression tests passed. TC-0901: Data export - PASSED.",
        "keyword_test/test_cases.md",
    )
    noise = _result("Operational incidents must be reviewed monthly.", "policy.md")

    result = apply_governance_cutoff(
        "Which document mentions TC-0901?",
        [exact_match, noise],
        _AbstainingGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [exact_match]
    assert result.metadata["stop_reason"] == "structured_lookup_exact_match"
    assert result.metadata["structured_lookup_contract"] == {
        "matched_identifiers": ["TC-0901"],
        "matched_sources": 1,
    }
    assert result.metadata["pyrrho"]["mode"] == "abstain"
    assert result.reasons == [
        "Structured lookup contract satisfied by exact identifier match: TC-0901."
    ]


class _TrustworthyGovernance:
    def decide(self, query: str, contexts: list[SimpleNamespace]) -> GovernanceDecision:
        return GovernanceDecision(
            mode=AnswerMode.TRUSTWORTHY,
            probs=(0.05, 0.05, 0.90),
            reason="Pyrrho: sources support a confident answer (P=0.90).",
        )


class _AbstainingGovernance:
    def decide(self, query: str, contexts: list[SimpleNamespace]) -> GovernanceDecision:
        return GovernanceDecision(
            mode=AnswerMode.ABSTAIN,
            probs=(0.80, 0.05, 0.15),
            reason="Pyrrho: retrieved sources do not contain enough evidence (P=0.80).",
        )


def _result(content: str, file_path: str) -> SimpleNamespace:
    """Build a compact ReadResult-like fixture."""
    return SimpleNamespace(
        content=content,
        file_path=file_path,
        address=SimpleNamespace(location=file_path, summary=content),
        metadata={},
    )


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

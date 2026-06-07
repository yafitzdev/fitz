# tests/unit/test_governance_cutoff.py
"""Unit tests for Pyrrho cutoff policy metadata."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.governance_cutoff import apply_governance_cutoff
from fitz_sage.governance.pyrrho import GovernanceDecision, HeadDecision


def test_cutoff_serializes_pyrrho_multitask_metadata():
    """Pyrrho head and scalar outputs should survive into evidence-pack metadata."""
    future_head = _head(
        raw_label="future_signal",
        final_label="future_signal",
        probabilities={"future_signal": 0.91, "other": 0.09},
    )
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
        retrieval_action=_head(
            raw_label="answer_now",
            final_label="answer_now",
            probabilities={"answer_now": 0.83, "retrieve_more": 0.17},
        ),
        gap_type=_head(
            raw_label="none",
            final_label="none",
            probabilities={"none": 0.86, "missing_specific_fact": 0.14},
        ),
        answerability_shape=_head(
            raw_label="direct_answer",
            final_label="direct_answer",
            probabilities={"direct_answer": 0.76, "structured_reasoning": 0.24},
        ),
        retrieval_modality=_head(
            raw_label="unstructured_text",
            final_label="unstructured_text",
            probabilities={"unstructured_text": 0.70, "structured_table": 0.30},
        ),
        heads={"future_head": future_head},
        scalars={
            "retrieval_retry_value": 0.17,
            "false_trustworthy_risk": 0.09,
            "evidence_failure_severity": 0.11,
        },
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
    assert pyrrho["retrieval_action"]["final_label"] == "answer_now"
    assert pyrrho["gap_type"]["final_label"] == "none"
    assert pyrrho["answerability_shape"]["final_label"] == "direct_answer"
    assert pyrrho["retrieval_modality"]["final_label"] == "unstructured_text"
    assert pyrrho["future_head"]["final_label"] == "future_signal"
    assert pyrrho["scalars"]["retrieval_retry_value"] == 0.17
    assert pyrrho["scalars"]["false_trustworthy_risk"] == 0.09
    assert pyrrho["scalars"]["evidence_failure_severity"] == 0.11


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


def test_retrieval_action_answer_now_allows_early_broad_stop():
    """A confident g4 answer_now action can override broad-shape minimum evidence."""
    results = [
        _result("The deployment guide says rollback requires `fitz deploy --rollback`.", "ops.md"),
        _result("Release notes mention deployment automation.", "release.md"),
        _result("The runbook lists monitoring alerts.", "alerts.md"),
        _result("The changelog records CLI changes.", "changelog.md"),
    ]
    decision = _decision(
        mode=AnswerMode.TRUSTWORTHY,
        action="answer_now",
        action_confidence=0.88,
    )

    result = apply_governance_cutoff(
        "What should I know about deployment rollback?",
        results,
        _FixedGovernance(decision),
        profile=SimpleNamespace(specificity="broad"),
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [results[0]]
    assert result.metadata["stop_reason"] == "pyrrho_answer_now"
    assert result.metadata["policy"]["query_shape"] == "broad"


def test_retrieval_action_retrieve_more_blocks_early_trustworthy_stop():
    """A confident retrieve_more action should force another prefix evaluation."""
    results = [
        _result("The summary says the invoice failed validation.", "summary.md"),
        _result("Invoice INV-17 failed retry validation in the audit log.", "audit.md"),
    ]

    result = apply_governance_cutoff(
        "Which invoice failed retry validation?",
        results,
        _PrefixGovernance(
            {
                1: _decision(
                    mode=AnswerMode.TRUSTWORTHY,
                    action="retrieve_more",
                    action_confidence=0.91,
                    gap="missing_specific_fact",
                ),
                2: _decision(
                    mode=AnswerMode.TRUSTWORTHY,
                    action="answer_now",
                    action_confidence=0.86,
                ),
            }
        ),
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == results
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert "retrieval_control_blocker" in result.metadata["trajectory"][0]
    assert result.metadata["trajectory"][0]["retrieval_action"]["final_label"] == "retrieve_more"


def test_retrieval_action_blocks_trustworthy_at_cutoff():
    """A final retrieve_more action should abstain instead of certifying weak evidence."""
    decision = _decision(
        mode=AnswerMode.TRUSTWORTHY,
        action="retrieve_more",
        action_confidence=0.91,
        gap="missing_specific_fact",
    )

    result = apply_governance_cutoff(
        "Which invoice failed retry validation?",
        [_result("The summary says an invoice failed validation.", "summary.md")],
        _FixedGovernance(decision),
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] == "retrieval_control_unsatisfied_at_cutoff"
    assert result.metadata["retrieval_control_blocker"] == (
        "Pyrrho retrieval action requested more evidence: retrieve_more."
    )
    assert result.metadata["trajectory"][0]["retrieval_control_blocker"] == (
        "Pyrrho retrieval action requested more evidence: retrieve_more."
    )


def test_retrieval_action_structured_lookup_satisfies_exact_identifier():
    """g4 structured_lookup can trigger exact-match cutoff without profile support."""
    exact_match = _result("The trace shows TC-0901 failed during export.", "trace.md")
    noise = _result("Other trace entries passed.", "trace.md")

    result = apply_governance_cutoff(
        "Which document mentions TC-0901?",
        [exact_match, noise],
        _FixedGovernance(
            _decision(
                mode=AnswerMode.ABSTAIN,
                action="structured_lookup",
                action_confidence=0.90,
                gap="missing_specific_fact",
            )
        ),
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [exact_match]
    assert result.metadata["stop_reason"] == "structured_lookup_exact_match"
    assert result.metadata["structured_lookup_contract"] == {
        "matched_identifiers": ["TC-0901"],
        "matched_sources": 1,
    }


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


def test_private_identifier_exact_match_satisfies_structured_lookup_contract():
    """Private Python function names should count as exact structured identifiers."""
    profile = SimpleNamespace(query_contract="structured_lookup")
    exact_match = _result(
        "def _format_query_profile(metadata: dict[str, object]) -> str:",
        "fitz_sage/cli/ui/display.py",
    )

    result = apply_governance_cutoff(
        "Which function named _format_query_profile formats query profile metadata?",
        [exact_match],
        _AbstainingGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [exact_match]
    assert result.metadata["stop_reason"] == "structured_lookup_exact_match"
    assert result.metadata["structured_lookup_contract"] == {
        "matched_identifiers": ["_format_query_profile"],
        "matched_sources": 1,
    }


def test_structured_lookup_does_not_match_identifier_as_loose_words():
    """Exact lookup should not confuse a function argument with another identifier."""
    profile = SimpleNamespace(query_contract="structured_lookup")
    near_match = _result(
        "def _format_query_profile(metadata: dict[str, object]) -> str:",
        "fitz_sage/cli/ui/display.py",
    )

    result = apply_governance_cutoff(
        "Which function named query_profile_metadata serializes Pyrrho query profile metadata?",
        [near_match],
        _AbstainingGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] != "structured_lookup_exact_match"


def test_contract_does_not_treat_natural_hyphen_term_as_exact_identifier():
    """Natural hyphenated prose should not become a hard exact-identifier requirement."""
    result = apply_governance_cutoff(
        "How does pre-retrieval planning work?",
        [_result("Planning runs before evidence retrieval.", "docs/planning.md")],
        _TrustworthyGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"


def test_comparison_contract_accepts_identifier_variant_without_trailing_topic():
    """Comparison entities with code identifiers should not require trailing topic nouns."""
    profile = SimpleNamespace(
        has_comparison_intent=True,
        comparison_entities=["query_profile_metadata", "_format_query_profile responsibilities"],
        answer_type="comparative",
    )
    metadata_result = _result(
        "def query_profile_metadata(query_signals, profile):",
        "fitz_sage/engines/fitz_krag/retrieval_profile.py",
    )
    display_result = _result(
        "def _format_query_profile(metadata: dict[str, object]) -> str:",
        "fitz_sage/cli/ui/display.py",
    )

    result = apply_governance_cutoff(
        "Compare query_profile_metadata and _format_query_profile responsibilities.",
        [metadata_result, display_result],
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [metadata_result, display_result]
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert "contract_blocker" not in result.metadata


def test_comparison_contract_blocks_answer_now_when_identifier_side_is_missing():
    """Pyrrho answer_now cannot certify one side of a comparison by loose word overlap."""
    profile = SimpleNamespace(
        has_comparison_intent=True,
        comparison_entities=["query_profile_metadata", "_format_query_profile responsibilities"],
        answer_type="comparative",
    )
    near_match = _result(
        "def _format_query_profile(metadata: dict[str, object]) -> str:",
        "fitz_sage/cli/ui/display.py",
    )

    result = apply_governance_cutoff(
        "Compare query_profile_metadata and _format_query_profile responsibilities.",
        [near_match],
        _FixedGovernance(
            _decision(
                mode=AnswerMode.TRUSTWORTHY,
                action="answer_now",
                action_confidence=0.95,
            )
        ),
        profile=profile,
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] == "contract_unsatisfied_at_cutoff"
    assert "query_profile_metadata" in result.reasons[0]


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


class _FixedGovernance:
    def __init__(self, decision: GovernanceDecision) -> None:
        self._decision = decision

    def decide(self, query: str, contexts: list[SimpleNamespace]) -> GovernanceDecision:
        return self._decision


class _PrefixGovernance:
    def __init__(self, decisions: dict[int, GovernanceDecision]) -> None:
        self._decisions = decisions

    def decide(self, query: str, contexts: list[SimpleNamespace]) -> GovernanceDecision:
        return self._decisions[len(contexts)]


def _result(content: str, file_path: str) -> SimpleNamespace:
    """Build a compact ReadResult-like fixture."""
    return SimpleNamespace(
        content=content,
        file_path=file_path,
        address=SimpleNamespace(location=file_path, summary=content),
        metadata={},
    )


def _decision(
    *,
    mode: AnswerMode,
    action: str | None = None,
    action_confidence: float = 0.0,
    gap: str = "none",
) -> GovernanceDecision:
    """Build a governance decision with optional g4 retrieval-control heads."""
    probs = {
        AnswerMode.ABSTAIN: (0.85, 0.05, 0.10),
        AnswerMode.DISPUTED: (0.05, 0.85, 0.10),
        AnswerMode.TRUSTWORTHY: (0.05, 0.10, 0.85),
    }[mode]
    action_other = "retrieve_more" if action == "answer_now" else "answer_now"
    retrieval_action = (
        _head(
            raw_label=action,
            final_label=action,
            probabilities={action: action_confidence, action_other: 1.0 - action_confidence},
        )
        if action
        else None
    )
    gap_other = "missing_specific_fact" if gap == "none" else "none"
    gap_type = _head(
        raw_label=gap,
        final_label=gap,
        probabilities={gap: 0.90, gap_other: 0.10},
    )
    heads = {"gap_type": gap_type}
    if retrieval_action is not None:
        heads["retrieval_action"] = retrieval_action
    return GovernanceDecision(
        mode=mode,
        probs=probs,
        reason=f"Pyrrho: {mode.value}.",
        retrieval_action=retrieval_action,
        gap_type=gap_type,
        heads=heads,
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

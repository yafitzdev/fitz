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


def test_pyrrho_controls_when_comparison_coverage_is_complete():
    """Comparison coverage is complete only when Pyrrho returns a trustworthy prefix."""
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
        _PrefixGovernance(
            {
                1: _decision(mode=AnswerMode.ABSTAIN),
                2: _decision(mode=AnswerMode.ABSTAIN),
                3: _decision(mode=AnswerMode.TRUSTWORTHY),
            }
        ),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert len(result.selected) == 3
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert "contract_blocker" not in result.metadata["trajectory"][1]


def test_pyrrho_abstention_is_final_at_cutoff():
    """Missing-evidence judgments come from Pyrrho, not from a sidecar contract gate."""
    profile = SimpleNamespace(has_temporal_intent=True, has_comparison_intent=False)

    result = apply_governance_cutoff(
        "What changed in March 2024?",
        [_result("April feedback improved after the beta release.", "feedback_april_2024.md")],
        _AbstainingGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] == "cutoff_exhausted"
    assert result.metadata["pyrrho"]["mode"] == "abstain"


def test_cutoff_does_not_override_pyrrho_trust_with_metric_gate():
    """The cutoff wrapper must not add its own metric sufficiency verdict."""
    profile = SimpleNamespace(has_temporal_intent=True, has_comparison_intent=False)

    result = apply_governance_cutoff(
        "What was Q4 2024 revenue?",
        [_result("Q4 2024 roadmap targets were postponed.", "roadmap_q4_2024.md")],
        _TrustworthyGovernance(),
        profile=profile,
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"
    assert "contract_blocker" not in result.metadata


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


def test_retrieval_action_structured_lookup_blocks_trust_without_pyrrho_trust():
    """structured_lookup is a Pyrrho retrieval-control request, not a trust verdict."""
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

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [exact_match, noise]
    assert result.metadata["stop_reason"] == "retrieval_control_unsatisfied_at_cutoff"
    assert "structured_lookup_contract" not in result.metadata


def test_structured_lookup_exact_identifier_does_not_override_pyrrho_abstention():
    """Source-finding exact matches are evidence for Pyrrho, not final governance."""
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

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [exact_match, noise]
    assert result.metadata["stop_reason"] == "cutoff_exhausted"
    assert "structured_lookup_contract" not in result.metadata
    assert result.metadata["pyrrho"]["mode"] == "abstain"


def test_private_identifier_exact_match_does_not_bypass_pyrrho():
    """Private Python function names cannot certify trust without Pyrrho."""
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

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [exact_match]
    assert result.metadata["stop_reason"] == "cutoff_exhausted"
    assert "structured_lookup_contract" not in result.metadata


def test_structured_lookup_does_not_match_identifier_as_loose_words():
    """Exact lookup matching is no longer a cutoff-level governance path."""
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
    """Pyrrho trustworthy verdicts still use comparison-shaped prefix sizing."""
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


def test_pyrrho_answer_now_is_authoritative_for_comparison_prefix():
    """A confident Pyrrho answer_now head is authoritative at cutoff time."""
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

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [near_match]
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"


def test_compiler_required_table_does_not_override_pyrrho_abstention():
    """Compiler-required table evidence is ledger input, not final governance."""
    table = _result(
        "Table: Warehouses\n"
        "warehouse_id | region | item | stock | unit\n"
        "WH-1 | west | flux capacitor | 17 | count",
        "structured/warehouses.csv",
    )
    table.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 4,
        "roles": ["required_table"],
        "min_sources": 1,
    }

    result = apply_governance_cutoff(
        "How many flux capacitor units are in the west region?",
        [table],
        _AbstainingGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [table]
    assert result.metadata["stop_reason"] == "cutoff_exhausted"


def test_evidence_contract_does_not_trust_conflict_roles():
    """A single compiler conflict role is not enough to force a dispute."""
    conflict = _result("Q1 revenue was 1.2 billion.", "finance.md")
    conflict.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 2,
        "roles": ["conflict_value"],
        "min_sources": 2,
    }

    result = apply_governance_cutoff(
        "What was Q1 revenue?",
        [conflict],
        _AbstainingGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.metadata["stop_reason"] == "cutoff_exhausted"


def test_pyrrho_dispute_owns_conflict_verdict():
    """Compiler-labeled conflicts need Pyrrho's DISPUTED verdict to govern."""
    code = _result("if user.get('archived') == 'true': return False", "feature_flags.py")
    code.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 5,
        "roles": ["conflict_value", "required_symbol"],
        "min_sources": 2,
    }
    docs = _result("Archived users remain eligible for beta flags.", "README.md")
    docs.metadata["evidence_compiler"] = {
        "rank": 2,
        "alignment_score": 4,
        "roles": ["conflict_value", "conflict_companion:documentation"],
        "min_sources": 2,
    }

    result = apply_governance_cutoff(
        "Are archived users eligible for beta feature flags?",
        [code, docs],
        _FixedGovernance(_decision(mode=AnswerMode.DISPUTED)),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.DISPUTED
    assert result.selected == [code, docs]
    assert result.metadata["stop_reason"] == "stable_dispute_at_cutoff"


def test_phrase_anchor_contract_does_not_override_pyrrho_abstention():
    """Phrase-anchor compiler roles cannot create a trustworthy result by themselves."""
    evidence = _result(
        "Enterprise Critical cases receive acknowledgement within 7 minutes.",
        "support_handbook.md",
    )
    evidence.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 6,
        "roles": ["anchor:Enterprise Critical"],
        "min_sources": 1,
    }

    result = apply_governance_cutoff(
        "What acknowledgement time applies to Enterprise Critical cases?",
        [evidence],
        _AbstainingGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [evidence]
    assert result.metadata["stop_reason"] == "cutoff_exhausted"


def test_compiler_min_sources_do_not_create_trustworthy_mode():
    """Compiler min_sources can assemble the prefix but cannot decide the mode."""
    table = _result("warehouse_id | item | stock\nWH-1 | flux capacitor | 17", "warehouses.csv")
    table.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 5,
        "roles": ["required_table"],
        "min_sources": 2,
    }
    release = _result("Release 2026.05 confirmed 17 flux capacitor units.", "release_notes.md")
    release.metadata["evidence_compiler"] = {
        "rank": 2,
        "alignment_score": 5,
        "roles": ["aligned"],
        "min_sources": 2,
    }

    result = apply_governance_cutoff(
        "Which release mentioned flux capacitor units and how many did it confirm?",
        [table, release],
        _AbstainingGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.ABSTAIN
    assert result.selected == [table, release]
    assert result.metadata["stop_reason"] == "cutoff_exhausted"


def test_compiler_min_sources_define_pyrrho_prefix_floor():
    """Pyrrho should not be asked to trust before the contract ledger bundle is visible."""
    table = _result("warehouse_id | item | stock\nWH-1 | flux capacitor | 17", "warehouses.csv")
    table.metadata["evidence_compiler"] = {
        "rank": 1,
        "alignment_score": 5,
        "roles": ["required_table"],
        "min_sources": 2,
    }
    release = _result("Release 2026.05 confirmed 17 flux capacitor units.", "release_notes.md")
    release.metadata["evidence_compiler"] = {
        "rank": 2,
        "alignment_score": 5,
        "roles": ["aligned"],
        "min_sources": 2,
    }

    result = apply_governance_cutoff(
        "Which release mentioned flux capacitor units and how many did it confirm?",
        [table, release],
        _TrustworthyGovernance(),
        profile=SimpleNamespace(),
    )

    assert result.mode is AnswerMode.TRUSTWORTHY
    assert result.selected == [table, release]
    assert result.metadata["trajectory"][0]["pyrrho_contract_prefix_min"] == 2
    assert result.metadata["stop_reason"] == "trustworthy_min_evidence_met"


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

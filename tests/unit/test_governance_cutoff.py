# tests/unit/test_governance_cutoff.py
"""Unit tests for the Pyrrho v2 cutoff policy."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.governance_cutoff import (
    apply_governance_cutoff,
    pyrrho_decision_metadata,
)
from fitz_sage.governance.pyrrho import (
    GovernanceDecision,
    HeadDecision,
    MultiLabelDecision,
)


def test_cutoff_serializes_only_native_pyrrho_v2_metadata():
    """Cutoff metadata should expose v2 heads and no removed legacy heads."""
    decision = _decision(AnswerMode.SUFFICIENT)
    result = apply_governance_cutoff(
        "What is the refund window?",
        [_result("Refund window is 30 days.")],
        _governance({1: decision}),
    )

    pyrrho = result.metadata["pyrrho"]
    assert result.mode is AnswerMode.SUFFICIENT
    assert pyrrho["mode"] == "sufficient"
    assert pyrrho["evidence_verdict"]["final_label"] == "SUFFICIENT"
    assert pyrrho["failure_mode"]["final_label"] == "none"
    assert pyrrho["retrieval_intents"]["final_labels"] == ["needs_lookup"]
    assert pyrrho["evidence_kinds"]["final_labels"] == ["needs_text"]
    assert ("retrieval" + "_action") not in pyrrho
    assert ("gap" + "_type") not in pyrrho
    assert "scalars" not in pyrrho


def test_pyrrho_decision_metadata_ignores_removed_heads_even_if_present():
    """External test doubles should not reintroduce removed Pyrrho metadata fields."""
    decision = _decision(AnswerMode.SUFFICIENT)
    removed_key = "retrieval" + "_action"
    object.__setattr__(decision, "heads", {**decision.heads, removed_key: _head("ready")})

    metadata = pyrrho_decision_metadata(AnswerMode.SUFFICIENT, decision)

    assert "evidence_verdict" in metadata
    assert removed_key not in metadata


def test_evidence_compiler_floor_delays_sufficient_stop():
    """Compiler-required sources can force Pyrrho to see enough ranked evidence."""
    results = [
        _result(
            "WH-1 row says 17 units.",
            index=1,
            kind="table",
            metadata={
                "evidence_compiler": {
                    "roles": ["required_table"],
                    "min_sources": 2,
                    "contract": {"required_modalities": ["table"]},
                }
            },
        ),
        _result("WH-1 table header confirms units.", index=2, kind="table"),
    ]

    result = apply_governance_cutoff(
        "How many units are in WH-1?",
        results,
        _governance(
            {
                1: _decision(AnswerMode.SUFFICIENT),
                2: _decision(AnswerMode.SUFFICIENT),
            }
        ),
    )

    assert result.mode is AnswerMode.SUFFICIENT
    assert result.metadata["evaluated"] == 2
    assert result.metadata["trajectory"][0]["evidence_prefix_min"] == 2


def test_bridge_evidence_delays_sufficient_stop_until_bridge_is_seen():
    """Pyrrho must inspect a retrieved bridge target before accepting sufficiency."""
    results = [
        _result("The glossary maps the service to SVC-202.", index=1),
        _result(
            "SVC-202 has a recovery objective of 15 minutes.",
            index=2,
            metadata={
                "evidence_compiler": {
                    "roles": ["bridge:SVC-202"],
                    "min_sources": 1,
                    "contract": {},
                }
            },
        ),
    ]

    result = apply_governance_cutoff(
        "What is the recovery objective for the mapped service?",
        results,
        _governance(
            {
                1: _decision(AnswerMode.SUFFICIENT),
                2: _decision(AnswerMode.SUFFICIENT),
            }
        ),
    )

    assert result.mode is AnswerMode.SUFFICIENT
    assert result.metadata["evaluated"] == 2
    assert result.metadata["trajectory"][0]["evidence_prefix_min"] == 2


def test_comparison_dispute_stops_when_pyrrho_verdict_policy_is_met():
    """Comparison queries should stop once Pyrrho gives a policy-eligible dispute."""
    profile = SimpleNamespace(query_contract="comparison_coverage")
    result = apply_governance_cutoff(
        "Compare refund windows in the old and new policies.",
        [
            _result("Old policy says 45 days.", index=1),
            _result("New policy says 30 days.", index=2),
        ],
        _governance(
            {
                1: _decision(AnswerMode.INSUFFICIENT),
                2: _decision(AnswerMode.DISPUTED),
            }
        ),
        profile=profile,
    )

    assert result.mode is AnswerMode.DISPUTED
    assert result.metadata["stop_reason"] == "disputed_min_evidence_met"
    assert result.metadata["evaluated"] == 2


def test_cutoff_returns_single_prefix_pyrrho_dispute_unchanged():
    """Fitz must not rewrite Pyrrho's only available verdict to insufficient."""
    result = apply_governance_cutoff(
        "What is the status of ECU-17A?",
        [_result("ECU-17A has two conflicting status values.")],
        _governance({1: _decision(AnswerMode.DISPUTED)}),
    )

    assert result.mode is AnswerMode.DISPUTED
    assert result.metadata["pyrrho"]["mode"] == "disputed"
    assert result.metadata["stop_reason"] == "disputed_min_evidence_met"


def test_cutoff_does_not_override_pyrrho_sufficient_with_local_conflict_signal():
    """Explicit-value conflicts are evidence for Pyrrho, not local verdict overrides."""
    profile = SimpleNamespace(query_contract="comparison_coverage")
    result = apply_governance_cutoff(
        "Do the policies agree on the refund window?",
        [
            _result("Legacy note says 45 days.", index=1),
            _result("Current policy says 30 days.", index=2),
        ],
        _governance(
            {
                1: _decision(AnswerMode.INSUFFICIENT),
                2: _decision(AnswerMode.SUFFICIENT),
            }
        ),
        profile=profile,
    )

    assert result.mode is AnswerMode.SUFFICIENT
    assert result.metadata["stop_reason"] == "sufficient_min_evidence_met"


def test_broad_overview_returns_representative_sources_without_pyrrho_call():
    """Representative overview is a retrieval surface, not a sufficiency verdict."""

    class _NoCallGovernance:
        supports_batched_prefixes = True

        def decide_many(self, query, prefixes):  # pragma: no cover - should not be called
            raise AssertionError("Pyrrho should not run for representative overview.")

    profile = SimpleNamespace(query_contract="representative_overview")
    results = [_result(f"Source {idx}.", index=idx) for idx in range(1, 9)]

    result = apply_governance_cutoff(
        "Summarize the docs.", results, _NoCallGovernance(), profile=profile
    )

    assert result.mode is AnswerMode.INSUFFICIENT
    assert result.metadata["stop_reason"] == "representative_overview"
    assert result.metadata["sufficiency_evaluated"] is False
    assert len(result.selected) == 6


def test_cutoff_exhaustion_returns_insufficient_without_old_control_reasons():
    """Cutoff exhaustion should report Pyrrho insufficiency, not retrieval-control state."""
    result = apply_governance_cutoff(
        "What is the launch date?",
        [_result("Only mentions launch owner.", index=1), _result("No date.", index=2)],
        _governance(
            {
                1: _decision(AnswerMode.INSUFFICIENT),
                2: _decision(AnswerMode.INSUFFICIENT),
            }
        ),
        requested_top_k=2,
    )

    assert result.mode is AnswerMode.INSUFFICIENT
    assert result.metadata["stop_reason"] == "cutoff_exhausted"
    assert ("retrieval" + "_control_blocker") not in result.metadata
    assert any("did not find sufficient" in reason for reason in result.reasons)


def _governance(decisions: dict[int, GovernanceDecision]):
    class _Governance:
        supports_batched_prefixes = True

        def decide_many(self, query, prefixes):
            return [decisions[len(prefix)] for prefix in prefixes]

    return _Governance()


def _decision(mode: AnswerMode) -> GovernanceDecision:
    labels = {
        AnswerMode.INSUFFICIENT: "INSUFFICIENT",
        AnswerMode.DISPUTED: "DISPUTED",
        AnswerMode.SUFFICIENT: "SUFFICIENT",
    }
    probs = {
        AnswerMode.INSUFFICIENT: (0.86, 0.08, 0.06),
        AnswerMode.DISPUTED: (0.07, 0.86, 0.07),
        AnswerMode.SUFFICIENT: (0.06, 0.08, 0.86),
    }[mode]
    evidence_verdict = _head(
        labels[mode],
        probabilities={
            "INSUFFICIENT": probs[0],
            "DISPUTED": probs[1],
            "SUFFICIENT": probs[2],
        },
    )
    failure_mode = _head("none")
    retrieval_intents = _multi_head("needs_lookup", ("needs_lookup",))
    evidence_kinds = _multi_head("needs_text", ("needs_text",))
    heads = {
        "evidence_verdict": evidence_verdict,
        "failure_mode": failure_mode,
        "retrieval_intents": retrieval_intents,
        "evidence_kinds": evidence_kinds,
    }
    return GovernanceDecision(
        mode=mode,
        probs=probs,
        reason=f"Pyrrho: {mode.value}.",
        evidence_verdict=evidence_verdict,
        failure_mode=failure_mode,
        retrieval_intents=retrieval_intents,
        evidence_kinds=evidence_kinds,
        heads=heads,
    )


def _head(label: str, *, probabilities: dict[str, float] | None = None) -> HeadDecision:
    probabilities = probabilities or {label: 0.91}
    return HeadDecision(
        raw_label=label,
        final_label=label,
        used_threshold_fallback=False,
        threshold=None,
        confidence=float(probabilities[label]),
        probabilities=probabilities,
        runner_up_label=label,
        runner_up_probability=float(probabilities[label]),
        margin_to_runner_up=0.0,
        entropy=0.0,
    )


def _multi_head(label: str, labels: tuple[str, ...]) -> MultiLabelDecision:
    probabilities = {item: (0.91 if item in labels else 0.05) for item in labels}
    return MultiLabelDecision(
        raw_label=label,
        final_label=label,
        final_labels=labels,
        used_threshold_fallback=False,
        threshold=0.5,
        confidence=0.91,
        probabilities=probabilities,
        runner_up_label=label,
        runner_up_probability=0.91,
        margin_to_runner_up=0.0,
        entropy=0.0,
    )


def _result(
    content: str,
    *,
    index: int = 1,
    kind: str = "section",
    metadata: dict | None = None,
) -> SimpleNamespace:
    address = SimpleNamespace(
        source_id=f"source-{index}",
        location=f"doc-{index}",
        summary="",
        kind=kind,
        metadata={},
        score=1.0,
    )
    return SimpleNamespace(
        content=content,
        file_path=f"doc-{index}.md",
        address=address,
        line_range=None,
        metadata=metadata or {},
    )

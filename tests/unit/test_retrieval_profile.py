# tests/unit/test_retrieval_profile.py
"""Unit tests for KRAG retrieval profile construction."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_sage.engines.fitz_krag.retrieval_profile import (
    build_retrieval_profile,
    query_profile_metadata,
)


def test_pyrrho_query_signals_steer_first_pass_profile():
    """Query-only Pyrrho heads should shape initial recall before evidence exists."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        query_contract=_head("comparison_coverage", 0.91),
        route=_head("law_policy", 0.88),
        answerability_shape=_head("structured_reasoning", 0.86),
        retrieval_modality=_head("structured_table", 0.90),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)

    assert profile.query_contract == "comparison_coverage"
    assert profile.query_route == "law_policy"
    assert profile.domain == "legal"
    assert profile.answerability_shape == "structured_reasoning"
    assert profile.retrieval_modality == "structured_table"
    assert profile.answer_type == "comparative"
    assert profile.has_comparison_intent is True
    assert profile.strategy_weights["table"] >= 0.55


def test_pyrrho_set_answer_broadens_first_pass_recall():
    """Set-shaped answers should broaden recall and mark aggregation intent."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        query_contract=_head("evidence_sufficiency", 0.90),
        answerability_shape=_head("set_answer", 0.89),
        retrieval_modality=_head("mixed", 0.84),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)

    assert profile.specificity == "broad"
    assert profile.answer_type == "exploratory"
    assert profile.has_aggregation_intent is True
    assert profile.entity_expansion_limit == 12
    assert profile.top_k > config.top_addresses
    assert profile.top_read > config.top_read
    assert profile.strategy_weights["code"] >= 0.30
    assert profile.strategy_weights["section"] >= 0.30
    assert profile.strategy_weights["table"] >= 0.25


def test_low_confidence_pyrrho_query_signals_still_own_retrieval():
    """Weak Pyrrho heads are exposed as uncertainty, not replaced by fitz-sage routing."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        query_contract=_head("evidence_sufficiency", 0.90),
        route=_head("law_policy", 0.59),
        answerability_shape=_head("set_answer", 0.59),
        retrieval_modality=_head("code", 0.59),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)

    assert profile.query_route == "law_policy"
    assert profile.query_route_confidence == 0.59
    assert profile.answerability_shape == "set_answer"
    assert profile.retrieval_modality == "code"
    assert profile.domain == "legal"
    assert profile.specificity == "broad"
    assert profile.answer_type == "exploratory"
    assert profile.strategy_weights["code"] >= 0.60


def test_pyrrho_obligation_forces_companion_evidence_modalities():
    """The obligation head is a retrieval contract signal, not debug-only metadata."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        retrieval_modality=_head("unstructured_text", 0.91),
        retrieval_obligation=_head("prose_plus_table", 0.21),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)
    metadata = query_profile_metadata(signals, profile)

    assert profile.retrieval_obligation == "prose_plus_table"
    assert profile.required_modalities == ("section", "table")
    assert profile.strategy_weights["section"] >= 0.45
    assert profile.strategy_weights["table"] >= 0.55
    assert metadata["signals"]["retrieval_obligation"]["used_for_retrieval"] is True


def test_v2_native_heads_steer_first_pass_profile():
    """v2 retrieval intents and evidence kinds should directly affect retrieval."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        retrieval_intents=_multi_head(
            ("needs_comparison_or_set", "needs_temporal_resolution"),
            {
                "needs_comparison_or_set": 0.91,
                "needs_temporal_resolution": 0.88,
                "needs_lookup": 0.20,
                "needs_broad_coverage": 0.10,
            },
        ),
        evidence_kinds=_multi_head(
            ("needs_table_or_record", "needs_text"),
            {
                "needs_table_or_record": 0.92,
                "needs_text": 0.77,
                "needs_code_or_symbol": 0.12,
            },
        ),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)
    metadata = query_profile_metadata(signals, profile)

    assert profile.retrieval_intents == (
        "needs_comparison_or_set",
        "needs_temporal_resolution",
    )
    assert profile.evidence_kinds == ("needs_table_or_record", "needs_text")
    assert profile.required_modalities == ("table", "section")
    assert profile.has_comparison_intent is True
    assert profile.has_temporal_intent is True
    assert profile.boost_recency is True
    assert profile.strategy_weights["table"] >= 0.55
    assert profile.strategy_weights["section"] >= 0.45
    assert metadata["signals"]["retrieval_intents"]["used_for_retrieval"] is True
    assert metadata["signals"]["evidence_kinds"]["final_labels"] == [
        "needs_table_or_record",
        "needs_text",
    ]


def test_pyrrho_mixed_modality_requires_all_evidence_kinds():
    """Mixed modality should make companion retrieval concrete."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        retrieval_modality=_head("mixed", 0.91),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)

    assert profile.required_modalities == ("section", "table", "symbol")
    assert profile.strategy_weights["section"] >= 0.45
    assert profile.strategy_weights["table"] >= 0.55
    assert profile.strategy_weights["code"] >= 0.60


def test_structured_lookup_keeps_code_search_eligible_for_data_phrasing():
    """Exact lookups should still search code when prose mentions table/section terms."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(primary_type=QueryType.DATA, confidence=0.80)
    signals = SimpleNamespace(query_contract=_head("structured_lookup", 0.90))

    profile = build_retrieval_profile(analysis, None, config, query_signals=signals)

    assert profile.query_contract == "structured_lookup"
    assert profile.strategy_weights["code"] > 0.05
    assert profile.strategy_weights["table"] >= 0.35


def test_inactive_v2_query_heads_use_deterministic_recall_fallback():
    """When v2 query heads are inactive, fallback planning should keep mixed evidence alive."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(
        primary_type=QueryType.GENERAL,
        confidence=0.65,
        refined_query=(
            "Using the export brief, should EXP-502 be skipped by the export scheduler?"
        ),
    )
    signals = SimpleNamespace(heads={})

    profile = build_retrieval_profile(analysis, None, config, query_signals=signals)

    assert profile.planning_owner == "deterministic_fallback"
    assert profile.query_contract == "comparison_coverage"
    assert profile.retrieval_modality == "mixed"
    assert profile.retrieval_obligation == "prose_plus_table"
    assert profile.required_modalities == ("section", "table", "symbol")
    assert profile.strategy_weights["section"] >= 0.45
    assert profile.strategy_weights["table"] >= 0.55
    assert profile.strategy_weights["code"] >= 0.60


def _head(label: str, confidence: float) -> SimpleNamespace:
    """Build a minimal Pyrrho head fixture."""
    return SimpleNamespace(
        final_label=label,
        confidence=confidence,
        probabilities={label: confidence},
    )


def _multi_head(
    labels: tuple[str, ...],
    probabilities: dict[str, float],
) -> SimpleNamespace:
    """Build a minimal Pyrrho multi-label head fixture."""
    return SimpleNamespace(
        final_label=labels[0],
        final_labels=labels,
        confidence=probabilities[labels[0]],
        probabilities=probabilities,
    )

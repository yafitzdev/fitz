# tests/unit/test_retrieval_profile.py
"""Unit tests for KRAG retrieval profile construction."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_sage.engines.fitz_krag.retrieval_profile import build_retrieval_profile


def test_pyrrho_query_signals_steer_first_pass_profile():
    """Query-only g4 heads should shape initial recall before evidence exists."""
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


def test_low_confidence_optional_query_signals_do_not_steer_retrieval():
    """Weak alpha heads should be recorded as uncertainty, not routing control."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    signals = SimpleNamespace(
        query_contract=_head("evidence_sufficiency", 0.90),
        route=_head("law_policy", 0.59),
        answerability_shape=_head("set_answer", 0.59),
        retrieval_modality=_head("code", 0.59),
    )

    profile = build_retrieval_profile(None, None, config, query_signals=signals)

    assert profile.query_route is None
    assert profile.query_route_confidence == 0.59
    assert profile.answerability_shape is None
    assert profile.retrieval_modality is None
    assert profile.domain == "general"
    assert profile.specificity == "moderate"
    assert profile.answer_type == "factual"
    assert profile.strategy_weights["code"] < 0.60


def test_structured_lookup_keeps_code_search_eligible_for_data_phrasing():
    """Exact lookups should still search code when prose mentions table/section terms."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(primary_type=QueryType.DATA, confidence=0.80)
    signals = SimpleNamespace(query_contract=_head("structured_lookup", 0.90))

    profile = build_retrieval_profile(analysis, None, config, query_signals=signals)

    assert profile.query_contract == "structured_lookup"
    assert profile.strategy_weights["code"] > 0.05
    assert profile.strategy_weights["table"] >= 0.35


def _head(label: str, confidence: float) -> SimpleNamespace:
    """Build a minimal Pyrrho head fixture."""
    return SimpleNamespace(
        final_label=label,
        confidence=confidence,
        probabilities={label: confidence},
    )

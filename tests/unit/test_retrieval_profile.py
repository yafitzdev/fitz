# tests/unit/test_retrieval_profile.py
"""Unit tests for KRAG retrieval profile construction."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.query_analyzer import QueryAnalysis, QueryType
from fitz_sage.engines.fitz_krag.retrieval_profile import (
    build_retrieval_profile,
    query_profile_metadata,
)


def test_data_lookup_uses_deterministic_structured_profile():
    """Exact data lookups should stay table-heavy without Pyrrho query heads."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(
        primary_type=QueryType.DATA,
        confidence=0.80,
        refined_query="How many units are listed for EXP-502 in the export table?",
    )

    profile = build_retrieval_profile(analysis, None, config)

    assert profile.planning_owner == "fitz_krag"
    assert profile.auxiliary_signal_policy == "deterministic_profile"
    assert profile.query_contract == "structured_lookup"
    assert profile.retrieval_modality == "structured_table"
    assert profile.required_modalities == ("table",)
    assert profile.strategy_weights["table"] >= 0.55
    assert profile.top_k >= config.top_addresses
    assert profile.top_read >= config.top_read


def test_mixed_evidence_query_keeps_all_required_modalities_alive():
    """Deterministic recall should preserve text/table/code evidence for mixed queries."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(
        primary_type=QueryType.GENERAL,
        confidence=0.65,
        refined_query=(
            "Using the export brief, should EXP-502 be skipped by the export scheduler?"
        ),
    )

    profile = build_retrieval_profile(analysis, None, config)

    assert profile.query_contract == "comparison_coverage"
    assert profile.retrieval_modality == "mixed"
    assert profile.retrieval_obligation == "prose_plus_table"
    assert profile.required_modalities == ("section", "table", "symbol")
    assert profile.has_comparison_intent is True
    assert profile.strategy_weights["section"] >= 0.45
    assert profile.strategy_weights["table"] >= 0.55
    assert profile.strategy_weights["code"] >= 0.60


def test_temporal_detection_boosts_recency():
    """Temporal retrieval is KRAG-owned and does not depend on Pyrrho query planning."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(
        primary_type=QueryType.DOCUMENTATION,
        confidence=0.75,
        refined_query="What is the latest refund policy?",
    )
    detection = SimpleNamespace(
        has_comparison_intent=False,
        has_temporal_intent=True,
        has_freshness_intent=True,
    )

    profile = build_retrieval_profile(analysis, detection, config)

    assert profile.query_contract == "temporal_grounding"
    assert profile.has_temporal_intent is True
    assert profile.boost_recency is True
    assert "needs_temporal_resolution" in profile.retrieval_intents


def test_extended_signals_still_adjust_profile_shape():
    """Local query intelligence can adjust retrieval shape without owning governance."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    analysis = QueryAnalysis(primary_type=QueryType.GENERAL, confidence=0.55)

    profile = build_retrieval_profile(
        analysis,
        None,
        config,
        extended_signals={
            "specificity": "broad",
            "answer_type": "exploratory",
            "domain": "technical",
        },
    )

    assert profile.specificity == "broad"
    assert profile.answer_type == "exploratory"
    assert profile.domain == "technical"
    assert profile.inject_corpus_summaries is True
    assert profile.entity_expansion_limit == 12
    assert profile.top_k > config.top_addresses


def test_pyrrho_pre_heads_own_retrieval_profile_when_available():
    """Pyrrho PRE labels should drive retrieval intent and evidence-surface knobs."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    pyrrho_plan = SimpleNamespace(
        retrieval_intents=SimpleNamespace(
            final_labels=("needs_comparison_or_set", "needs_temporal_resolution"),
            final_label="needs_comparison_or_set",
            confidence=0.91,
            probabilities={
                "needs_comparison_or_set": 0.91,
                "needs_temporal_resolution": 0.73,
            },
        ),
        evidence_kinds=SimpleNamespace(
            final_labels=("needs_text", "needs_table_or_record"),
            final_label="needs_text",
            confidence=0.88,
            probabilities={"needs_text": 0.88, "needs_table_or_record": 0.84},
        ),
    )

    profile = build_retrieval_profile(
        QueryAnalysis(
            primary_type=QueryType.GENERAL,
            confidence=0.70,
            refined_query="Compare current release notes against the rollout table.",
        ),
        None,
        config,
        pyrrho_plan=pyrrho_plan,
    )
    metadata = query_profile_metadata(profile, pyrrho_plan)

    assert profile.planning_owner == "pyrrho"
    assert profile.auxiliary_signal_policy == "pyrrho_v2_pre_with_deterministic_fallback"
    assert profile.query_contract == "temporal_grounding"
    assert profile.retrieval_modality == "mixed"
    assert profile.retrieval_obligation == "prose_plus_table"
    assert profile.required_modalities == ("section", "table", "symbol")
    assert metadata["pyrrho_pre"]["retrieval_intents"]["final_labels"] == [
        "needs_comparison_or_set",
        "needs_temporal_resolution",
    ]


def test_query_profile_metadata_has_no_external_signal_section():
    """Profile metadata should not serialize removed Pyrrho query-head signals."""
    config = SimpleNamespace(top_addresses=20, top_read=10)
    profile = build_retrieval_profile(
        QueryAnalysis(
            primary_type=QueryType.DATA,
            confidence=0.80,
            refined_query="Show record EXP-502.",
        ),
        None,
        config,
    )

    metadata = query_profile_metadata(profile)

    assert set(metadata) == {"profile"}
    assert "signals" not in metadata
    assert metadata["profile"]["planning_owner"] == "fitz_krag"
    assert metadata["profile"]["auxiliary_signal_policy"] == "deterministic_profile"

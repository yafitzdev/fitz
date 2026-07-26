# tests/unit/test_query_planner.py
"""Tests for deterministic KRAG query planning."""

from __future__ import annotations

from fitz_sage.engines.fitz_krag.query_analyzer import QueryType
from fitz_sage.engines.fitz_krag.query_planner import DeterministicQueryPlanner
from fitz_sage.retrieval.detection.modules import AggregationType, TemporalIntent


class TestDeterministicQueryPlanner:
    """No-chat planner behavior."""

    def test_plans_comparison_temporal_and_keywords_without_llm(self):
        """Planner should surface retrieval signals without a chat provider."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("Compare Q1 2024 vs Q2 2024 API failures")

        assert plan.analysis.primary_type == QueryType.CODE
        assert plan.detection is not None
        assert plan.detection.has_temporal_intent
        assert plan.detection.temporal_intent == TemporalIntent.POINT_IN_TIME
        assert plan.detection.has_comparison_intent
        assert plan.detection.comparison_queries
        assert "API" in plan.keywords
        assert "failures" in plan.keywords
        assert plan.extended_signals == {
            "specificity": "moderate",
            "answer_type": "comparative",
            "domain": "technical",
        }

    def test_plans_aggregation_fetch_multiplier(self):
        """Aggregation cues should increase fetch breadth through DetectionSummary."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("How many failed test cases were reported?")

        assert plan.detection is not None
        assert plan.detection.has_aggregation_intent
        assert plan.detection.aggregation_type == AggregationType.COUNT
        assert plan.detection.fetch_multiplier == 4

    def test_plans_explicit_set_quantifiers_as_aggregation(self):
        """Common exhaustive-set language should request broad retrieval."""
        planner = DeterministicQueryPlanner()

        for query in (
            "Find every environment variable.",
            "Show the owner for each vendor.",
            "Return the full set of dependencies.",
        ):
            plan = planner.plan(query)
            assert plan.detection is not None
            assert plan.detection.has_aggregation_intent

        plan = planner.plan("List the key metrics in the review.")
        assert plan.detection is not None
        assert plan.detection.has_aggregation_intent

    def test_scalar_measurement_is_not_a_corpus_aggregation(self):
        """A singular duration lookup should remain a narrow fact request."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("How many days is the token rotation interval?")

        assert plan.detection is not None
        assert not plan.detection.has_aggregation_intent

    def test_key_facts_query_is_broad_exploratory(self):
        """Corpus key-facts queries should use the broad cutoff policy."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("What are the key facts in this corpus?")

        assert plan.extended_signals["specificity"] == "broad"
        assert plan.extended_signals["answer_type"] == "exploratory"

    def test_respects_disabled_detection(self):
        """Detection can be disabled while analysis and keywords still work."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("latest API errors", detection_enabled=False)

        assert plan.analysis.primary_type == QueryType.CODE
        assert plan.detection is None
        assert "errors" in plan.keywords

    def test_detects_changed_between_temporal_comparison(self):
        """Between-period change wording should route like a comparison."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("What changed between Q1 and Q2 2024?")

        assert plan.detection is not None
        assert plan.detection.has_temporal_intent
        assert plan.detection.has_comparison_intent
        assert plan.detection.comparison_entities == ["Q1", "Q2 2024"]

    def test_temporal_detection_recognizes_month_names(self):
        """Month-scoped queries should carry temporal references."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("What changed in March 2024?")

        assert plan.detection is not None
        assert plan.detection.has_temporal_intent
        assert "march 2024" in plan.detection.temporal.metadata["references"]

    def test_temporal_detection_recognizes_lifecycle_language(self):
        """Version-state and relative-time language should retain temporal intent."""
        planner = DeterministicQueryPlanner()

        for query in (
            "Which retention rule is effective now?",
            "What was the original deadline?",
            "Who owned the account at that time?",
        ):
            plan = planner.plan(query)
            assert plan.detection is not None
            assert plan.detection.has_temporal_intent

    def test_release_version_does_not_imply_temporal_intent(self):
        """A dotted release id is an identifier unless the query adds time scope."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("Which region uses release 2026.05?")

        assert plan.detection is not None
        assert not plan.detection.has_temporal_intent

    def test_exact_identifier_keywords_remain_literal(self):
        """Deterministic planning must not invent separator variants for IDs."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("Find TC_1000")

        assert "TC_1000" in plan.keywords
        assert "TC-1000" not in plan.keywords
        assert "TC 1000" not in plan.keywords

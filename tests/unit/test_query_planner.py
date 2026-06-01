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
        assert "endpoint" in plan.keywords
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

    def test_respects_disabled_detection(self):
        """Detection can be disabled while analysis and keywords still work."""
        planner = DeterministicQueryPlanner()

        plan = planner.plan("latest API errors", detection_enabled=False)

        assert plan.analysis.primary_type == QueryType.CODE
        assert plan.detection is None
        assert "failure" in plan.keywords

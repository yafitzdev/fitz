# tests/test_constraints.py
"""
Tests for the Constraint Plugin system.

These tests verify:
1. ConflictAwareConstraint detects contradictions
2. Resolution queries bypass conflict detection
3. Pipeline integration works correctly

Uses semantic matching with mock embedder for testing.
"""

from __future__ import annotations

import pytest

from fitz_ai.core.chunk import Chunk
from fitz_ai.core.guardrails import (
    ConflictAwareConstraint,
    ConstraintResult,
    SemanticMatcher,
    apply_constraints,
)
from tests.conftest_guardrails import create_deterministic_embedder

# =============================================================================
# Test Data
# =============================================================================


def make_chunk(id: str, content: str) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=id,
        doc_id=f"doc_{id}",
        content=content,
        chunk_index=0,
        metadata={},
    )


@pytest.fixture
def semantic_matcher() -> SemanticMatcher:
    """Create a semantic matcher with mock embedder for testing."""
    embedder = create_deterministic_embedder()
    return SemanticMatcher(
        embedder=embedder,
        causal_threshold=0.70,
        assertion_threshold=0.70,
        query_threshold=0.70,
        conflict_threshold=0.70,
    )


# =============================================================================
# Tests: ConstraintResult
# =============================================================================


class TestConstraintResult:
    """Test ConstraintResult factory methods."""

    def test_allow_factory(self):
        result = ConstraintResult.allow()
        assert result.allow_decisive_answer is True
        assert result.reason is None

    def test_deny_factory(self):
        result = ConstraintResult.deny("test reason", key="value")
        assert result.allow_decisive_answer is False
        assert result.reason == "test reason"
        assert result.metadata["key"] == "value"


# =============================================================================
# Tests: ConflictAwareConstraint
# =============================================================================


class TestConflictAwareConstraint:
    """Test the default conflict detection constraint."""

    def test_no_conflict_allows_answer(self, semantic_matcher):
        """Should allow decisive answer when no conflicts."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "Incident 17B was a security incident."),
            make_chunk("2", "The security incident caused significant downtime."),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is True

    def test_detects_security_vs_operational_conflict(self, semantic_matcher):
        """Should detect conflicting security vs operational classifications."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "Incident 17B was a security incident with unauthorized access."),
            make_chunk(
                "2", "Incident 17B was an operational incident with misconfigured settings."
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is False

    def test_empty_chunks_allows_answer(self, semantic_matcher):
        """Should allow when no chunks retrieved."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        result = constraint.apply("Any question?", [])

        assert result.allow_decisive_answer is True

    def test_disabled_constraint_always_allows(self, semantic_matcher):
        """Should allow when constraint is disabled."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher, enabled=False)

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_resolution_query_allows_despite_conflict(self, semantic_matcher):
        """Should allow decisive answer when query asks for resolution."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "Source A says: security incident."),
            make_chunk("2", "Source B says: operational incident."),
        ]

        # Explicitly asking for resolution
        result = constraint.apply(
            "Which classification should be considered authoritative, and why?",
            chunks,
        )

        assert result.allow_decisive_answer is True

    def test_detects_trend_conflict_improved_vs_declined(self, semantic_matcher):
        """Should detect conflicting trend claims."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "Customer satisfaction improved significantly this quarter."),
            make_chunk("2", "Customer satisfaction declined compared to last year."),
        ]

        result = constraint.apply("How did customer satisfaction change?", chunks)

        assert result.allow_decisive_answer is False

    def test_detects_sentiment_conflict(self, semantic_matcher):
        """Should detect conflicting sentiment claims."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "The overall feedback was positive."),
            make_chunk("2", "Customer response was negative."),
        ]

        result = constraint.apply("What was the customer feedback?", chunks)

        assert result.allow_decisive_answer is False

    def test_detects_state_conflict_successful_vs_failed(self, semantic_matcher):
        """Should detect conflicting state claims."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "The deployment was successful and completed."),
            make_chunk("2", "The deployment failed with errors."),
        ]

        result = constraint.apply("Was the deployment successful?", chunks)

        assert result.allow_decisive_answer is False

    def test_no_conflict_for_agreeing_trends(self, semantic_matcher):
        """Should not flag conflict when trends agree."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "Sales improved this quarter."),
            make_chunk("2", "Revenue also improved significantly."),
        ]

        result = constraint.apply("How did the business perform?", chunks)

        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Constraint Runner
# =============================================================================


class TestApplyConstraints:
    """Test the constraint runner."""

    def test_no_constraints_allows(self):
        """Should allow when no constraints configured."""
        result = apply_constraints("query", [], [])
        assert result.allow_decisive_answer is True

    def test_single_constraint_deny(self, semantic_matcher):
        """Should deny when single constraint denies."""
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk("1", "This is a security incident with unauthorized access."),
            make_chunk("2", "This is an operational incident from misconfiguration."),
        ]

        result = apply_constraints("What type?", chunks, [constraint])

        assert result.allow_decisive_answer is False

    def test_multiple_constraints_any_deny(self):
        """Should deny if any constraint denies."""

        class AlwaysAllowConstraint:
            name = "always_allow"

            def apply(self, query, chunks):
                return ConstraintResult.allow()

        class AlwaysDenyConstraint:
            name = "always_deny"

            def apply(self, query, chunks):
                return ConstraintResult.deny("test denial")

        constraints = [AlwaysAllowConstraint(), AlwaysDenyConstraint()]

        result = apply_constraints("query", [], constraints)

        assert result.allow_decisive_answer is False
        assert "test denial" in result.reason

    def test_constraint_exception_continues(self):
        """Should continue if a constraint raises an exception."""

        class CrashingConstraint:
            name = "crasher"

            def apply(self, query, chunks):
                raise RuntimeError("Oops!")

        class WorkingConstraint:
            name = "working"

            def apply(self, query, chunks):
                return ConstraintResult.allow()

        constraints = [CrashingConstraint(), WorkingConstraint()]

        # Should not raise, should allow (fail-safe)
        result = apply_constraints("query", [], constraints)

        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Acceptance Criteria (from handoff)
# =============================================================================


class TestAcceptanceCriteria:
    """
    Tests matching the acceptance criteria from the handoff document.

    These are the key behavioral tests that must pass.
    """

    def test_contradiction_query_must_surface_disagreement(self, semantic_matcher):
        """
        Acceptance test: Contradiction detection.

        Query: "Was Incident 17B a security incident?"

        With conflicting docs, the constraint MUST deny decisive answer.
        The system should then surface disagreement (handled by generation).
        """
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk(
                "doc_a",
                "Incident 17B Investigation Report: This was a security incident "
                "involving unauthorized access attempts.",
            ),
            make_chunk(
                "doc_b",
                "Incident 17B Post-Mortem: This was an operational incident "
                "with misconfigured firewall settings.",
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        # MUST deny decisive answer when conflicts exist
        assert result.allow_decisive_answer is False
        assert result.reason is not None

    def test_authority_resolution_query_allowed(self, semantic_matcher):
        """
        Acceptance test: Authority resolution.

        Query: "Which classification should be considered authoritative, and why?"

        Even with conflicting docs, this query explicitly asks for resolution,
        so the constraint should allow a decisive answer.
        """
        constraint = ConflictAwareConstraint(semantic_matcher=semantic_matcher)

        chunks = [
            make_chunk(
                "doc_a",
                "Security team classified Incident 17B as a security incident.",
            ),
            make_chunk(
                "doc_b",
                "Operations team logged Incident 17B as an operational incident.",
            ),
        ]

        result = constraint.apply(
            "Which classification should be considered authoritative, and why?",
            chunks,
        )

        # MUST allow decisive answer for resolution queries
        assert result.allow_decisive_answer is True

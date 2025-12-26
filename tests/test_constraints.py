# tests/test_constraints.py
"""
Tests for the Constraint Plugin system.

These tests verify:
1. ConflictAwareConstraint detects contradictions
2. Resolution queries bypass conflict detection
3. Pipeline integration works correctly
"""

from __future__ import annotations

from fitz_ai.engines.classic_rag.constraints import (
    ConflictAwareConstraint,
    ConstraintResult,
    apply_constraints,
)
from fitz_ai.engines.classic_rag.models.chunk import Chunk

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

    def test_no_conflict_allows_answer(self):
        """Should allow decisive answer when no conflicts."""
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk("1", "Incident 17B was a security incident."),
            make_chunk("2", "The security incident caused significant downtime."),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is True

    def test_detects_security_vs_operational_conflict(self):
        """Should detect conflicting security vs operational classifications."""
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk("1", "Incident 17B was classified as a security incident."),
            make_chunk(
                "2",
                "Incident 17B was an operational incident caused by misconfiguration.",
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        assert result.allow_decisive_answer is False
        assert "security" in result.reason.lower()
        assert "operational" in result.reason.lower()

    def test_detects_explicit_classification_conflict(self):
        """Should detect explicitly stated classification conflicts."""
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk("1", "Type: Security Incident"),
            make_chunk("2", "Categorized as: Operational Incident"),
        ]

        result = constraint.apply("What type of incident was this?", chunks)

        assert result.allow_decisive_answer is False

    def test_empty_chunks_allows_answer(self):
        """Should allow when no chunks retrieved."""
        constraint = ConflictAwareConstraint()

        result = constraint.apply("Any question?", [])

        assert result.allow_decisive_answer is True

    def test_disabled_constraint_always_allows(self):
        """Should allow when constraint is disabled."""
        constraint = ConflictAwareConstraint(enabled=False)

        chunks = [
            make_chunk("1", "This is a security incident."),
            make_chunk("2", "This is an operational incident."),
        ]

        result = constraint.apply("What type?", chunks)

        assert result.allow_decisive_answer is True

    def test_resolution_query_allows_despite_conflict(self):
        """Should allow decisive answer when query asks for resolution."""
        constraint = ConflictAwareConstraint()

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

    def test_why_disagree_query_allows(self):
        """Should allow when asking why sources disagree."""
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk("1", "Security incident report."),
            make_chunk("2", "Operational incident log."),
        ]

        result = constraint.apply(
            "Why do the documents disagree about the classification?",
            chunks,
        )

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

    def test_single_constraint_deny(self):
        """Should deny when single constraint denies."""
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk("1", "Security incident."),
            make_chunk("2", "Operational incident."),
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

    def test_contradiction_query_must_surface_disagreement(self):
        """
        Acceptance test: Contradiction detection.

        Query: "Was Incident 17B a security incident?"

        With conflicting docs, the constraint MUST deny decisive answer.
        The system should then surface disagreement (handled by generation).
        """
        constraint = ConflictAwareConstraint()

        chunks = [
            make_chunk(
                "doc_a",
                "Incident 17B Investigation Report: This was determined to be "
                "a security incident involving unauthorized access attempts.",
            ),
            make_chunk(
                "doc_b",
                "Incident 17B Post-Mortem: Root cause was an operational incident "
                "caused by a misconfigured firewall rule during maintenance.",
            ),
        ]

        result = constraint.apply("Was Incident 17B a security incident?", chunks)

        # MUST deny decisive answer when conflicts exist
        assert result.allow_decisive_answer is False
        assert result.reason is not None

    def test_authority_resolution_query_allowed(self):
        """
        Acceptance test: Authority resolution.

        Query: "Which classification should be considered authoritative, and why?"

        Even with conflicting docs, this query explicitly asks for resolution,
        so the constraint should allow a decisive answer.
        """
        constraint = ConflictAwareConstraint()

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

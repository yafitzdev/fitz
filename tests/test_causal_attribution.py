# tests/engines/classic_rag/constraints/test_causal_attribution.py
"""
Tests for CausalAttributionConstraint.

These tests verify:
1. Causal queries without causal evidence are denied
2. Causal queries with explicit causal language are allowed
3. Non-causal queries pass through
"""

from __future__ import annotations

import pytest

from fitz_ai.engines.classic_rag.constraints import CausalAttributionConstraint
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
# Tests: Basic Behavior
# =============================================================================


class TestBasicBehavior:
    """Test basic constraint behavior."""

    def test_non_causal_query_allowed(self):
        """Non-causal queries should pass through."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "Helios was deprecated in 2023.")]

        result = constraint.apply("What is Helios?", chunks)

        assert result.allow_decisive_answer is True

    def test_disabled_always_allows(self):
        """Should allow when disabled."""
        constraint = CausalAttributionConstraint(enabled=False)

        chunks = [make_chunk("1", "Helios was deprecated.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_empty_chunks_defers(self):
        """Empty chunks should defer to other constraints."""
        constraint = CausalAttributionConstraint()

        result = constraint.apply("Why did X happen?", [])

        # Defers to InsufficientEvidenceConstraint
        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Causal Query Detection
# =============================================================================


class TestCausalQueryDetection:
    """Test detection of causal queries."""

    @pytest.mark.parametrize(
        "query",
        [
            "Why was Helios deprecated?",
            "Why did the system fail?",
            "What caused the outage?",
            "What led to the incident?",
            "How come the service crashed?",
            "Explain the failure",
            "What is the reason for the change?",
        ],
    )
    def test_detects_causal_queries(self, query: str):
        """Should detect causal queries."""
        constraint = CausalAttributionConstraint()

        # No causal evidence
        chunks = [make_chunk("1", "The system was deprecated.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is False

    @pytest.mark.parametrize(
        "query",
        [
            "What is Helios?",
            "When was Helios deprecated?",
            "Who maintains the system?",
            "List all incidents",
            "Describe the architecture",
        ],
    )
    def test_non_causal_queries_pass(self, query: str):
        """Non-causal queries should pass through."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "The system was deprecated.")]

        result = constraint.apply(query, chunks)

        assert result.allow_decisive_answer is True


# =============================================================================
# Tests: Causal Evidence Detection
# =============================================================================


class TestCausalEvidenceDetection:
    """Test detection of causal language in evidence."""

    @pytest.mark.parametrize(
        "causal_phrase",
        [
            "because of high costs",
            "due to reliability issues",
            "caused by a bug",
            "led to the outage",
            "resulted in data loss",
            "as a result of the incident",
            "therefore we changed",
            "thus the system failed",
            "attributed to human error",
            "triggered by the update",
        ],
    )
    def test_allows_with_causal_markers(self, causal_phrase: str):
        """Should allow when causal language is present."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", f"Helios was deprecated {causal_phrase}.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_denies_without_causal_markers(self):
        """Should deny when no causal language is present."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "Helios was deprecated in August 2023."),
            make_chunk("2", "The system is no longer maintained."),
            make_chunk("3", "Orion replaced Helios as primary system."),
        ]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is False
        assert "causal" in result.reason.lower()

    def test_signal_is_qualified_not_abstain(self):
        """Signal should be 'qualified' not 'abstain' (we have evidence, just not causal)."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "Helios was deprecated.")]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.signal == "qualified"


# =============================================================================
# Tests: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases."""

    def test_causal_marker_in_any_chunk_allows(self):
        """Should allow if ANY chunk has causal language."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "Helios was deprecated in 2023."),
            make_chunk("2", "This was due to high operational costs."),  # Has causal
            make_chunk("3", "Orion is now primary."),
        ]

        result = constraint.apply("Why was Helios deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_case_insensitive_detection(self):
        """Should detect causal markers case-insensitively."""
        constraint = CausalAttributionConstraint()

        chunks = [make_chunk("1", "BECAUSE OF HIGH COSTS, we deprecated it.")]

        result = constraint.apply("Why was it deprecated?", chunks)

        assert result.allow_decisive_answer is True

    def test_metadata_includes_chunk_counts(self):
        """Denial should include chunk count metadata."""
        constraint = CausalAttributionConstraint()

        chunks = [
            make_chunk("1", "System A was deprecated."),
            make_chunk("2", "System B replaced it."),
        ]

        result = constraint.apply("Why was System A deprecated?", chunks)

        assert result.metadata.get("total_chunks") == 2
        assert result.metadata.get("causal_chunks") == 0

# tests/core/test_answer_mode.py
"""
Tests for the AnswerMode system.

These tests verify:
1. AnswerMode enum values
2. AnswerModeResolver logic
3. Mode instruction mapping
"""

from __future__ import annotations

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.core.answer_mode_resolver import resolve_answer_mode
from fitz_ai.engines.classic_rag.constraints.base import ConstraintResult
from fitz_ai.engines.classic_rag.generation.answer_mode.instructions import (
    MODE_INSTRUCTIONS,
    get_mode_instruction,
)

# =============================================================================
# Tests: AnswerMode Enum
# =============================================================================


class TestAnswerModeEnum:
    """Test AnswerMode enum."""

    def test_all_modes_defined(self):
        """Should have all expected modes."""
        assert AnswerMode.CONFIDENT == "confident"
        assert AnswerMode.QUALIFIED == "qualified"
        assert AnswerMode.DISPUTED == "disputed"
        assert AnswerMode.ABSTAIN == "abstain"

    def test_mode_is_string(self):
        """Modes should be string-compatible."""
        assert isinstance(AnswerMode.CONFIDENT, str)
        assert AnswerMode.CONFIDENT.value == "confident"


# =============================================================================
# Tests: AnswerModeResolver
# =============================================================================


class TestAnswerModeResolver:
    """Test the answer mode resolution logic."""

    def test_empty_results_returns_confident(self):
        """Should return CONFIDENT when no constraint results."""
        mode = resolve_answer_mode([])
        assert mode == AnswerMode.CONFIDENT

    def test_all_allowed_returns_confident(self):
        """Should return CONFIDENT when all constraints allow."""
        results = [
            ConstraintResult.allow(),
            ConstraintResult.allow(),
        ]
        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.CONFIDENT

    def test_abstain_signal_returns_abstain(self):
        """Should return ABSTAIN when abstain signal present."""
        results = [
            ConstraintResult.allow(),
            ConstraintResult.deny("No evidence", signal="abstain"),
        ]
        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.ABSTAIN

    def test_disputed_signal_returns_disputed(self):
        """Should return DISPUTED when disputed signal present."""
        results = [
            ConstraintResult.allow(),
            ConstraintResult.deny("Conflicting sources", signal="disputed"),
        ]
        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.DISPUTED

    def test_abstain_takes_priority_over_disputed(self):
        """ABSTAIN should take priority over DISPUTED."""
        results = [
            ConstraintResult.deny("Conflict", signal="disputed"),
            ConstraintResult.deny("No evidence", signal="abstain"),
        ]
        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.ABSTAIN

    def test_denial_without_signal_returns_qualified(self):
        """Should return QUALIFIED when denied without signal."""
        results = [
            ConstraintResult.deny("Some issue"),  # No signal
        ]
        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.QUALIFIED


# =============================================================================
# Tests: Mode Instructions
# =============================================================================


class TestModeInstructions:
    """Test mode instruction mapping."""

    def test_all_modes_have_instructions(self):
        """Every mode should have an instruction."""
        for mode in AnswerMode:
            assert mode in MODE_INSTRUCTIONS
            assert len(MODE_INSTRUCTIONS[mode]) > 0

    def test_get_mode_instruction(self):
        """get_mode_instruction should return correct instruction."""
        instruction = get_mode_instruction(AnswerMode.DISPUTED)
        assert "disagree" in instruction.lower()

        instruction = get_mode_instruction(AnswerMode.ABSTAIN)
        assert "definitive" in instruction.lower()

    def test_confident_instruction_is_direct(self):
        """CONFIDENT instruction should be direct."""
        instruction = get_mode_instruction(AnswerMode.CONFIDENT)
        assert "clearly" in instruction.lower() or "directly" in instruction.lower()


# =============================================================================
# Tests: Integration (Constraint → Mode)
# =============================================================================


class TestConstraintToModeIntegration:
    """Test the full flow from constraint results to mode."""

    def test_conflict_constraint_produces_disputed(self):
        """ConflictAwareConstraint denial should produce DISPUTED mode."""
        # Simulate what ConflictAwareConstraint returns
        result = ConstraintResult.deny(
            reason="Conflicting classifications detected",
            signal="disputed",
        )

        mode = resolve_answer_mode([result])
        assert mode == AnswerMode.DISPUTED

    def test_insufficient_evidence_produces_abstain(self):
        """InsufficientEvidenceConstraint denial should produce ABSTAIN mode."""
        # Simulate what InsufficientEvidenceConstraint returns
        result = ConstraintResult.deny(
            reason="No explicit causal evidence found",
            signal="abstain",
        )

        mode = resolve_answer_mode([result])
        assert mode == AnswerMode.ABSTAIN

    def test_both_constraints_deny_abstain_wins(self):
        """When both constraints deny, ABSTAIN should take priority."""
        results = [
            ConstraintResult.deny("Conflict", signal="disputed"),
            ConstraintResult.deny("No evidence", signal="abstain"),
        ]

        mode = resolve_answer_mode(results)
        assert mode == AnswerMode.ABSTAIN

# tests/unit/test_answer_mode.py
"""Tests for the AnswerMode enum + mode-instruction mapping.

The governance decision layer is Pyrrho;
end-to-end behaviour is exercised via the smoke test, not here.
"""

from __future__ import annotations

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.generation.mode_instructions import (
    MODE_INSTRUCTIONS,
    get_mode_instruction,
)


class TestAnswerModeEnum:
    def test_all_modes_defined(self):
        assert AnswerMode.SUFFICIENT == "sufficient"
        assert AnswerMode.DISPUTED == "disputed"
        assert AnswerMode.INSUFFICIENT == "insufficient"

    def test_mode_is_string(self):
        assert isinstance(AnswerMode.SUFFICIENT, str)
        assert AnswerMode.SUFFICIENT.value == "sufficient"


class TestModeInstructions:
    def test_all_modes_have_instructions(self):
        for mode in AnswerMode:
            assert mode in MODE_INSTRUCTIONS
            assert len(MODE_INSTRUCTIONS[mode]) > 0

    def test_get_mode_instruction(self):
        instruction = get_mode_instruction(AnswerMode.DISPUTED)
        assert "disagree" in instruction.lower()

        instruction = get_mode_instruction(AnswerMode.INSUFFICIENT)
        assert "definitive" in instruction.lower()

    def test_sufficient_instruction_is_direct(self):
        instruction = get_mode_instruction(AnswerMode.SUFFICIENT)
        assert "clearly" in instruction.lower() or "directly" in instruction.lower()

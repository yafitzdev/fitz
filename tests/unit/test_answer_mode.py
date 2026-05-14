# tests/unit/test_answer_mode.py
"""Tests for the AnswerMode enum + mode-instruction mapping.

The governance decision layer is now pyrrho (`fitz_sage.governance.pyrrho`);
end-to-end behaviour is exercised via the smoke test, not here.
"""

from __future__ import annotations

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.instructions import (
    MODE_INSTRUCTIONS,
    get_mode_instruction,
)


class TestAnswerModeEnum:
    def test_all_modes_defined(self):
        assert AnswerMode.TRUSTWORTHY == "trustworthy"
        assert AnswerMode.DISPUTED == "disputed"
        assert AnswerMode.ABSTAIN == "abstain"

    def test_mode_is_string(self):
        assert isinstance(AnswerMode.TRUSTWORTHY, str)
        assert AnswerMode.TRUSTWORTHY.value == "trustworthy"


class TestModeInstructions:
    def test_all_modes_have_instructions(self):
        for mode in AnswerMode:
            assert mode in MODE_INSTRUCTIONS
            assert len(MODE_INSTRUCTIONS[mode]) > 0

    def test_get_mode_instruction(self):
        instruction = get_mode_instruction(AnswerMode.DISPUTED)
        assert "disagree" in instruction.lower()

        instruction = get_mode_instruction(AnswerMode.ABSTAIN)
        assert "definitive" in instruction.lower()

    def test_trustworthy_instruction_is_direct(self):
        instruction = get_mode_instruction(AnswerMode.TRUSTWORTHY)
        assert "clearly" in instruction.lower() or "directly" in instruction.lower()

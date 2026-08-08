"""Map an already-decided answer mode into synthesis instructions."""

from fitz_sage.core.answer_mode import AnswerMode

MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    AnswerMode.SUFFICIENT: "Answer clearly and directly based on the evidence.",
    AnswerMode.DISPUTED: (
        "State explicitly that sources disagree and summarize the disagreement. "
        "Do not assert one view as correct."
    ),
    AnswerMode.INSUFFICIENT: (
        "State that the available information does not allow a definitive answer. "
        "Do not guess or invent explanations."
    ),
}


def get_mode_instruction(mode: AnswerMode) -> str:
    """Return the synthesis instruction for Pyrrho's mapped verdict."""
    return MODE_INSTRUCTIONS[mode]


__all__ = ["MODE_INSTRUCTIONS", "get_mode_instruction"]

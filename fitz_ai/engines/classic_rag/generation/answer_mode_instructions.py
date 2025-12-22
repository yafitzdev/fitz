# fitz_ai/engines/classic_rag/generation/answer_mode_instructions.py
"""
Answer Mode Instructions - Maps AnswerMode to synthesis instructions.

These instructions are prepended to the RGS prompt to control
the epistemic posture of the generated answer.

The LLM does not reason about mode selection - it just follows
the instruction it receives.
"""

from fitz_ai.core.answer_mode import AnswerMode


MODE_INSTRUCTIONS: dict[AnswerMode, str] = {
    AnswerMode.CONFIDENT: (
        "Answer clearly and directly based on the evidence."
    ),
    AnswerMode.QUALIFIED: (
        "Answer carefully and note any uncertainty or limitations in the evidence."
    ),
    AnswerMode.DISPUTED: (
        "State explicitly that sources disagree and summarize the disagreement. "
        "Do not assert one view as correct."
    ),
    AnswerMode.ABSTAIN: (
        "State that the available information does not allow a definitive answer. "
        "Do not guess or invent explanations."
    ),
}


def get_mode_instruction(mode: AnswerMode) -> str:
    """
    Get the synthesis instruction for an answer mode.

    Args:
        mode: The AnswerMode to get instruction for

    Returns:
        Instruction string to prepend to the prompt
    """
    return MODE_INSTRUCTIONS.get(mode, MODE_INSTRUCTIONS[AnswerMode.CONFIDENT])
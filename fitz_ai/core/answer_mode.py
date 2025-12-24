# fitz_ai/core/answer_mode.py
"""
Answer Mode - Epistemic framing for answers.

AnswerMode controls how certain the answer should sound,
not what it says. It is determined by constraint signals
after retrieval, before synthesis.

Modes:
- CONFIDENT: Answer clearly and directly
- QUALIFIED: Note uncertainty or limitations
- DISPUTED: Explicitly state sources disagree
- ABSTAIN: State that evidence is insufficient
"""

from enum import Enum


class AnswerMode(str, Enum):
    """
    Epistemic posture for answer generation.

    Selected based on constraint signals, not LLM reasoning.
    """

    CONFIDENT = "confident"
    """Evidence supports a clear, direct answer."""

    QUALIFIED = "qualified"
    """Answer with noted uncertainty or limitations."""

    DISPUTED = "disputed"
    """Sources explicitly disagree; summarize the disagreement."""

    ABSTAIN = "abstain"
    """Evidence is insufficient; do not attempt a definitive answer."""

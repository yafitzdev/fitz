# fitz_sage/core/answer_mode.py
"""
Answer Mode - Epistemic framing for answers.

AnswerMode controls how certain the answer should sound,
not what it says. It is determined by constraint signals
after retrieval, before synthesis.

Modes:
- SUFFICIENT: Answer clearly and directly based on the evidence
- DISPUTED: Explicitly state sources disagree
- INSUFFICIENT: State that evidence is insufficient
"""

from enum import Enum


class AnswerMode(str, Enum):
    """
    Epistemic posture for answer generation.

    Selected based on constraint signals, not LLM reasoning.
    """

    SUFFICIENT = "sufficient"
    """Evidence supports answering. Answer clearly and directly."""

    DISPUTED = "disputed"
    """Sources explicitly disagree; summarize the disagreement."""

    INSUFFICIENT = "insufficient"
    """Evidence is insufficient; do not attempt a definitive answer."""

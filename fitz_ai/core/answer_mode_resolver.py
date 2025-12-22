# fitz_ai/core/answer_mode_resolver.py
"""
Answer Mode Resolver - Maps constraint signals to epistemic posture.

This is a single, deterministic function that decides how certain
the answer should sound based on constraint results.

Rules (in priority order):
1. If any constraint signals "abstain" → ABSTAIN
2. If any constraint signals "disputed" → DISPUTED
3. If any constraint denied (without specific signal) → QUALIFIED
4. Otherwise → CONFIDENT
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from fitz_ai.core.answer_mode import AnswerMode

if TYPE_CHECKING:
    from fitz_ai.engines.classic_rag.constraints.base import ConstraintResult


def resolve_answer_mode(
    results: Sequence["ConstraintResult"],
) -> AnswerMode:
    """
    Resolve the answer mode from constraint results.

    Args:
        results: List of ConstraintResult from applied constraints

    Returns:
        AnswerMode indicating epistemic posture for the answer

    Rules (deterministic, ordered):
    - "abstain" signal → ABSTAIN (highest priority)
    - "disputed" signal → DISPUTED
    - Any denial without signal → QUALIFIED
    - All allowed → CONFIDENT
    """
    if not results:
        return AnswerMode.CONFIDENT

    # Collect signals from all results
    signals = {r.signal for r in results if r.signal}

    # Priority 1: Abstain (insufficient evidence)
    if "abstain" in signals:
        return AnswerMode.ABSTAIN

    # Priority 2: Disputed (conflicting sources)
    if "disputed" in signals:
        return AnswerMode.DISPUTED

    # Priority 3: Any denial without specific signal
    if any(not r.allow_decisive_answer for r in results):
        return AnswerMode.QUALIFIED

    # Default: Confident
    return AnswerMode.CONFIDENT
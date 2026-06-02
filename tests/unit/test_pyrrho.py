# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import GovernanceDecision, Pyrrho, _head_decision


def test_decide_many_batches_non_empty_prefixes():
    """decide_many should batch non-empty prefixes and preserve empty-prefix abstain."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.ABSTAIN,
                probs=(0.9, 0.05, 0.05),
                reason="Need more evidence.",
            ),
            GovernanceDecision(
                mode=AnswerMode.TRUSTWORTHY,
                probs=(0.05, 0.05, 0.9),
                reason="Enough evidence.",
            ),
        ]
    )

    decisions = pyrrho.decide_many(
        "What happened?",
        [
            [],
            [SimpleNamespace(content="No answer here.")],
            [
                SimpleNamespace(content="The release shipped."),
                SimpleNamespace(content="The changelog confirms it."),
            ],
        ],
    )

    assert [decision.mode for decision in decisions] == [
        AnswerMode.ABSTAIN,
        AnswerMode.ABSTAIN,
        AnswerMode.TRUSTWORTHY,
    ]
    pyrrho._predict_context_batches.assert_called_once_with(
        "What happened?",
        [
            ["No answer here."],
            ["The release shipped.", "The changelog confirms it."],
        ],
    )


def test_head_decision_threshold_falls_back_from_weak_trustworthy():
    """Weak TRUSTWORTHY predictions should expose the threshold fallback."""
    decision = _head_decision(
        torch.tensor([0.1, 0.2, 0.25]),
        {0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"},
        trustworthy_threshold=0.5,
    )

    assert decision.raw_label == "TRUSTWORTHY"
    assert decision.final_label == "DISPUTED"
    assert decision.used_threshold_fallback is True
    assert decision.threshold == 0.5
    assert decision.runner_up_label == "TRUSTWORTHY"
    assert decision.margin_to_runner_up < 0.0
    assert decision.probabilities.keys() == {"ABSTAIN", "DISPUTED", "TRUSTWORTHY"}

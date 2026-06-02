# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import Pyrrho


def test_decide_many_batches_non_empty_prefixes():
    """decide_many should tokenize/run non-empty prefixes as one batch."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._encode = MagicMock(return_value={"input_ids": np.array([[1], [2]])})
    pyrrho._run = MagicMock(
        return_value=np.array(
            [
                [3.0, 0.0, 0.0],
                [0.0, 0.0, 3.0],
            ]
        )
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
    pyrrho._encode.assert_called_once()
    assert len(pyrrho._encode.call_args.args[0]) == 2
    pyrrho._run.assert_called_once()

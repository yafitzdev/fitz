# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import Pyrrho


def test_decide_many_runs_non_empty_prefixes_serially():
    """decide_many should avoid batch execution for the current ONNX export."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._encode = MagicMock(return_value={"input_ids": np.array([[1]])})
    pyrrho._run = MagicMock(
        side_effect=[
            np.array([[3.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 3.0]]),
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
    assert pyrrho._encode.call_count == 2
    assert pyrrho._run.call_count == 2
    assert isinstance(pyrrho._encode.call_args_list[0].args[0], str)

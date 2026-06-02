# tests/unit/test_encoder_smoke.py
"""
Smoke test — actually load and run the local managed classifiers.

The rest of the unit suite never triggers a real model load: both
encoders lazy-import their dependencies inside `_load()`, so the suite
only ever checks that the provider objects *construct*. This test
closes that gap — it downloads (first run) and loads the Pyrrho
governance classifier and the ONNX gte-reranker cross-encoder, then runs
one real forward pass each.

Marked `slow` (network + model load); excluded from `pytest -m "not
slow"`. This is the coverage that catches a fresh-install crash on the
first encoder call.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


def test_pyrrho_loads_and_decides():
    """create_governance('pyrrho') -> a real local Pyrrho decide() call."""
    from fitz_sage.core.answer_mode import AnswerMode
    from fitz_sage.governance import create_governance

    governance = create_governance("pyrrho")
    assert governance is not None

    contexts = [SimpleNamespace(content="The capital of France is Paris.", metadata={})]
    decision = governance.decide("What is the capital of France?", contexts)

    assert decision.mode in (
        AnswerMode.TRUSTWORTHY,
        AnswerMode.DISPUTED,
        AnswerMode.ABSTAIN,
    )
    assert len(decision.probs) == 3
    assert abs(sum(decision.probs) - 1.0) < 1e-3  # softmax distribution


def test_onnx_reranker_loads_and_ranks():
    """OnnxReranker() -> a real INT8 ONNX cross-encoder forward pass."""
    from fitz_sage.llm.providers.onnx_reranker import OnnxReranker

    reranker = OnnxReranker()
    results = reranker.rerank(
        "battery warranty period",
        ["The battery warranty is 8 years.", "Charging the battery takes 45 minutes."],
    )

    assert len(results) == 2
    assert results[0].score >= results[1].score  # sorted by relevance
    assert results[0].index == 0  # the warranty doc outranks the charging doc

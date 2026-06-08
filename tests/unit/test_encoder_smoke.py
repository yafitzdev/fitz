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

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.slow


def _pyrrho_smoke_spec() -> str:
    """Use the local canonical g4-alpha package when this checkout has it."""
    env_path = os.environ.get("PYRRHO_G4_ALPHA_PACKAGE")
    if env_path and Path(env_path).is_dir():
        return f"pyrrho/{env_path}"

    repo_root = Path(__file__).resolve().parents[2]
    sibling_package = repo_root.parent / "pyrrho" / "models" / "pyrrho-nano-g4-alpha"
    if sibling_package.is_dir():
        return f"pyrrho/{sibling_package}"

    return "pyrrho"


def test_pyrrho_loads_and_decides():
    """Canonical Pyrrho g4-alpha package -> a real local decide() call."""
    from fitz_sage.core.answer_mode import AnswerMode
    from fitz_sage.governance import create_governance

    governance = create_governance(_pyrrho_smoke_spec())
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

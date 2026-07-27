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
    """Use the local canonical v2 package when this checkout has it."""
    env_path = os.environ.get("PYRRHO_V2_PACKAGE")
    if env_path and Path(env_path).is_dir():
        return f"pyrrho/{env_path}"

    repo_root = Path(__file__).resolve().parents[2]
    sibling_package = repo_root.parent / "pyrrho" / "models" / "pyrrho-v2-nano-g1"
    if sibling_package.is_dir():
        return f"pyrrho/{sibling_package}"

    return "pyrrho"


def test_pyrrho_loads_and_decides():
    """Canonical Pyrrho v2 package -> a real local decide() call."""
    from fitz_sage.integrations.pyrrho import create_pyrrho

    pyrrho = create_pyrrho(_pyrrho_smoke_spec())

    contexts = [SimpleNamespace(content="The capital of France is Paris.", metadata={})]
    decision = pyrrho.decide("What is the capital of France?", contexts)

    assert decision.verdict in {"SUFFICIENT", "DISPUTED", "INSUFFICIENT"}
    assert set(decision.probabilities) == {"SUFFICIENT", "DISPUTED", "INSUFFICIENT"}
    assert abs(sum(decision.probabilities.values()) - 1.0) < 1e-3


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

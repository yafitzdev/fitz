# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho v2 governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import (
    PYRRHO_POST_TAG,
    PYRRHO_PRE_TAG,
    GovernanceDecision,
    MultiLabelDecision,
    Pyrrho,
    PyrrhoQueryPlan,
    _format_input,
    _format_query_input,
    _is_v2_model_dir,
    _load_sufficient_threshold,
    _preferred_onnx_path,
    _v2_decision_from_logits,
)


def test_decide_many_batches_non_empty_prefixes():
    """decide_many should batch non-empty prefixes and preserve empty-prefix insufficient."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.INSUFFICIENT,
                probs=(0.9, 0.05, 0.05),
                reason="Need more evidence.",
            ),
            GovernanceDecision(
                mode=AnswerMode.SUFFICIENT,
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
        AnswerMode.INSUFFICIENT,
        AnswerMode.INSUFFICIENT,
        AnswerMode.SUFFICIENT,
    ]
    pyrrho._predict_context_batches.assert_called_once_with(
        "What happened?",
        [
            ["No answer here."],
            ["The release shipped.", "The changelog confirms it."],
        ],
    )


def test_decide_many_passes_only_evidence_text_to_model():
    """Compiler and closure metadata must not be injected into Pyrrho input."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.SUFFICIENT,
                probs=(0.05, 0.05, 0.9),
                reason="Enough evidence.",
            )
        ]
    )

    pyrrho.decide_many(
        "How many units are in WH-1?",
        [
            [
                SimpleNamespace(
                    content="WH-1 | west | flux capacitor | 17",
                    metadata={
                        "evidence_compiler": {
                            "roles": ["required_table"],
                            "min_sources": 1,
                        },
                        "evidence_closure": {
                            "role": "required_table",
                            "reason": "missing_table_modality",
                        },
                    },
                )
            ]
        ],
    )

    contexts = pyrrho._predict_context_batches.call_args.args[1]
    assert contexts[0][0] == "WH-1 | west | flux capacitor | 17"
    assert "evidence_compiler" not in contexts[0][0]
    assert "evidence_closure" not in contexts[0][0]


def test_v2_decision_exposes_only_native_heads():
    """v2 decisions should not synthesize removed legacy heads."""
    logits = np.asarray(
        [
            0.0,
            0.0,
            4.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            -3.0,
            -3.0,
            -3.0,
            3.0,
            3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
        ],
        dtype=np.float32,
    )

    decision = _v2_decision_from_logits(logits)

    assert decision.mode is AnswerMode.SUFFICIENT
    assert decision.probs[2] > 0.9
    assert decision.evidence_verdict is not None
    assert decision.evidence_verdict.final_label == "SUFFICIENT"
    assert decision.failure_mode is not None
    assert decision.failure_mode.final_label == "none"
    assert decision.retrieval_intents is not None
    assert isinstance(decision.retrieval_intents, MultiLabelDecision)
    assert decision.evidence_kinds is not None
    assert decision.evidence_kinds.final_labels == (
        "needs_text",
        "needs_table_or_record",
    )
    assert set(decision.heads) == {
        "evidence_verdict",
        "failure_mode",
        "retrieval_intents",
        "evidence_kinds",
    }
    assert not hasattr(decision, "retrieval" + "_action")
    assert not hasattr(decision, "gap" + "_type")


def test_pyrrho_pre_and_post_inputs_use_mode_tags():
    """Dual v2 packages require explicit pre/post mode tags."""
    assert _format_query_input("What changed?").startswith(PYRRHO_PRE_TAG)
    assert _format_input("What changed?", ["Release notes say v2 shipped."]).startswith(
        PYRRHO_POST_TAG
    )


def test_plan_query_returns_only_pre_retrieval_heads():
    """The query-only pass exposes retrieval/evidence heads, not post verdicts."""
    logits = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0,
            -3.0,
            3.0,
            -3.0,
            3.0,
            3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
        ],
        dtype=np.float32,
    )
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._run_onnx_texts = MagicMock(return_value=np.asarray([logits]))

    plan = pyrrho.plan_query("Compare release notes and rollout table")

    assert isinstance(plan, PyrrhoQueryPlan)
    assert plan.retrieval_intents.final_labels == (
        "needs_lookup",
        "needs_comparison_or_set",
    )
    assert plan.evidence_kinds.final_labels == (
        "needs_text",
        "needs_table_or_record",
    )
    pyrrho._run_onnx_texts.assert_called_once_with(
        [f"{PYRRHO_PRE_TAG}\nQuestion: Compare release notes and rollout table"]
    )


def test_v2_threshold_falls_back_from_weak_sufficient():
    """Packaged thresholds should protect weak SUFFICIENT predictions."""
    logits = np.asarray(
        [
            0.0,
            0.1,
            0.2,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
        ],
        dtype=np.float32,
    )

    decision = _v2_decision_from_logits(logits, sufficient_threshold=0.5)

    assert decision.evidence_verdict is not None
    assert decision.evidence_verdict.raw_label == "SUFFICIENT"
    assert decision.evidence_verdict.final_label == "DISPUTED"
    assert decision.evidence_verdict.used_threshold_fallback is True
    assert decision.mode is AnswerMode.DISPUTED


def test_load_sufficient_threshold_reads_manifest(tmp_path):
    """Packaged release thresholds should override the default constant."""
    (tmp_path / "manifest.json").write_text(
        '{"release": {"sufficient_threshold": 0.44}}',
        encoding="utf-8",
    )

    assert _load_sufficient_threshold(tmp_path) == 0.44


def test_preferred_onnx_path_prefers_fp32_graph(tmp_path):
    """v2 packages should prefer the FP32 ONNX graph for governance accuracy."""
    fp32 = tmp_path / "model.onnx"
    quantized = tmp_path / "model_quantized.onnx"
    fp32.write_bytes(b"fp32")
    quantized.write_bytes(b"int8")

    assert _preferred_onnx_path(tmp_path) == fp32


def test_run_onnx_texts_feeds_only_graph_declared_numpy_inputs():
    """ONNX inference should feed only graph-declared NumPy inputs."""

    class _Input:
        def __init__(self, name: str) -> None:
            self.name = name

    class _Session:
        def __init__(self) -> None:
            self.feed = None

        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

        def run(self, output_names, feed):
            self.feed = feed
            return [np.asarray([[0.0] * 18], dtype=np.float32)]

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            assert kwargs["return_tensors"] == "np"
            return {
                "input_ids": np.asarray([[1, 2]], dtype=np.int64),
                "attention_mask": np.asarray([[1, 1]], dtype=np.int64),
                "token_type_ids": np.asarray([[0, 0]], dtype=np.int64),
            }

    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._tokenizer = _Tokenizer()
    pyrrho._model = _Session()

    logits = pyrrho._run_onnx_texts(["Question: q\n\nSources:\na"])

    assert logits.shape == (1, 18)
    assert set(pyrrho._model.feed) == {"input_ids", "attention_mask"}


def test_load_requires_v2_package_with_onnx(tmp_path):
    """Pyrrho no longer loads packages without the native v2 ONNX contract."""
    (tmp_path / "config.json").write_text('{"id2label": {"0": "OTHER"}}', encoding="utf-8")
    pyrrho = Pyrrho(model_id=str(tmp_path))

    with pytest.raises(ValueError, match="v2 package"):
        pyrrho._load()


def test_is_v2_model_dir_recognizes_native_labels(tmp_path):
    """The package validator should recognize the native v2 label layout."""
    labels = {
        "0": "evidence_verdict.INSUFFICIENT",
        "1": "evidence_verdict.DISPUTED",
        "2": "evidence_verdict.SUFFICIENT",
        "3": "failure_mode.none",
        "8": "retrieval_intents.needs_lookup",
        "12": "evidence_kinds.needs_text",
    }
    (tmp_path / "config.json").write_text(
        '{"id2label": ' + repr(labels).replace("'", '"') + "}",
        encoding="utf-8",
    )

    assert _is_v2_model_dir(tmp_path) is True

# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import (
    GovernanceDecision,
    MultiLabelDecision,
    Pyrrho,
    QueryDecision,
    _decision_from_outputs,
    _head_decision,
    _load_trustworthy_threshold,
    _preferred_onnx_path,
    _required_label_map,
    _v2_decision_from_logits,
    _v2_query_decision_from_logits,
)


def test_decide_many_batches_non_empty_prefixes():
    """decide_many should batch non-empty prefixes and preserve empty-prefix abstain."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._model_kind = "g5"
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


def test_decide_many_passes_evidence_ledger_to_model():
    """Compiler and closure metadata should be visible to Pyrrho as evidence text."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._model_kind = "g5"
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.TRUSTWORTHY,
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
                            "contract": {
                                "required_modalities": ["table"],
                                "identifiers": ["WH-1"],
                            },
                        },
                        "evidence_closure": {
                            "role": "required_table",
                            "reason": "missing_table_modality",
                            "bridges": ["WH-1"],
                        },
                    },
                )
            ]
        ],
    )

    contexts = pyrrho._predict_context_batches.call_args.args[1]
    serialized = contexts[0][0]
    assert "Pyrrho evidence ledger:" in serialized
    assert "compiler roles: required_table" in serialized
    assert "compiler minimum sources: 1" in serialized
    assert "contract required_modalities: table" in serialized
    assert "closure role: required_table" in serialized


def test_v2_decide_many_omits_runtime_evidence_ledger():
    """v2 was trained on source text, not fitz-sage's runtime ledger."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._model_kind = "v2_alpha"
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.TRUSTWORTHY,
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
                            "contract": {
                                "required_modalities": ["table"],
                                "identifiers": ["WH-1"],
                            },
                        },
                    },
                )
            ]
        ],
    )

    contexts = pyrrho._predict_context_batches.call_args.args[1]
    serialized = contexts[0][0]
    assert serialized == "WH-1 | west | flux capacitor | 17"
    assert "Pyrrho evidence ledger:" not in serialized


def test_head_decision_threshold_falls_back_from_weak_trustworthy():
    """Weak TRUSTWORTHY predictions should expose the threshold fallback."""
    decision = _head_decision(
        np.asarray([0.1, 0.2, 0.25], dtype=np.float32),
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


def test_decision_from_outputs_requires_g5_heads():
    """g5 heads should survive conversion into GovernanceDecision."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._id2label = {0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"}
    pyrrho._query_contract_id2label = {0: "evidence_sufficiency", 1: "structured_lookup"}
    pyrrho._route_id2label = {0: "technology_computing", 1: "general_commonsense"}
    pyrrho._taxonomy_id2label = {0: "direct_answer", 1: "evidence_absent"}
    pyrrho._g5_head_id2labels = {
        "retrieval_action": {0: "answer_now", 1: "retrieve_more"},
        "gap_type": {0: "none", 1: "missing_specific_fact"},
        "answerability_shape": {0: "direct_answer", 1: "structured_reasoning"},
        "retrieval_modality": {0: "unstructured_text", 1: "structured_table"},
        "retrieval_obligation": {0: "row_key_lookup", 1: "multi_row_comparison"},
    }
    pyrrho._scalar_fields = ("evidence_sufficiency", "evidence_failure_severity")
    pyrrho._trustworthy_threshold = 0.34
    outputs = {
        "governance_logits": np.asarray([[0.0, 0.1, 2.0]], dtype=np.float32),
        "query_contract_logits": np.asarray([[0.0, 1.0]], dtype=np.float32),
        "route_logits": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "taxonomy_logits": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "retrieval_action_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "gap_type_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "answerability_shape_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "retrieval_modality_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "retrieval_obligation_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        "scalar_preds": np.asarray([[0.75, 0.25]], dtype=np.float32),
    }

    decision = _decision_from_outputs(pyrrho, outputs, 0)

    assert decision.mode is AnswerMode.TRUSTWORTHY
    assert decision.governance.threshold == 0.34
    assert decision.query_contract.final_label == "structured_lookup"
    assert decision.retrieval_action.final_label == "retrieve_more"
    assert decision.gap_type.final_label == "missing_specific_fact"
    assert decision.answerability_shape.final_label == "structured_reasoning"
    assert decision.retrieval_modality.final_label == "structured_table"
    assert decision.retrieval_obligation.final_label == "multi_row_comparison"
    assert decision.heads["retrieval_action"] is decision.retrieval_action
    assert decision.heads["retrieval_obligation"] is decision.retrieval_obligation
    assert decision.scalars == {
        "evidence_sufficiency": 0.75,
        "evidence_failure_severity": 0.25,
    }


def test_predict_query_includes_query_only_heads():
    """classify_query should return the full pre-retrieval Pyrrho query shape."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._run_batch = MagicMock(
        return_value={
            "query_contract_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
            "route_logits": np.asarray([[2.0, 0.0]], dtype=np.float32),
            "answerability_shape_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
            "retrieval_modality_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
            "retrieval_obligation_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
            "retrieval_action_logits": np.asarray([[0.0, 2.0]], dtype=np.float32),
        }
    )
    pyrrho._query_contract_id2label = {0: "evidence_sufficiency", 1: "structured_lookup"}
    pyrrho._route_id2label = {0: "technology_computing", 1: "general_commonsense"}
    pyrrho._g5_head_id2labels = {
        "retrieval_action": {0: "answer_now", 1: "retrieve_more"},
        "answerability_shape": {0: "direct_answer", 1: "structured_reasoning"},
        "retrieval_modality": {0: "unstructured_text", 1: "structured_table"},
        "retrieval_obligation": {0: "row_key_lookup", 1: "multi_row_comparison"},
    }

    decision = pyrrho.classify_query("Which benchmark row is lower?")

    assert isinstance(decision, QueryDecision)
    assert decision.query_contract.final_label == "structured_lookup"
    assert decision.route.final_label == "technology_computing"
    assert decision.answerability_shape.final_label == "structured_reasoning"
    assert decision.retrieval_modality.final_label == "structured_table"
    assert decision.retrieval_obligation.final_label == "multi_row_comparison"
    assert decision.heads["query_contract"] is decision.query_contract
    assert decision.heads["retrieval_obligation"] is decision.retrieval_obligation
    assert "retrieval_action" not in decision.heads


def test_v2_decision_exposes_only_native_heads():
    """v2 decisions should not synthesize legacy Pyrrho heads."""
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

    assert decision.mode is AnswerMode.TRUSTWORTHY
    assert decision.probs[2] > 0.9
    assert decision.governance is None
    assert decision.query_contract is None
    assert decision.route is None
    assert decision.taxonomy is None
    assert decision.retrieval_action is None
    assert decision.gap_type is None
    assert decision.answerability_shape is None
    assert decision.retrieval_modality is None
    assert decision.retrieval_obligation is None
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
    assert decision.heads["evidence_verdict"] is decision.evidence_verdict
    assert decision.scalars == {}


def test_v2_query_decision_is_inactive_until_query_head_is_trained():
    """v2 should not emit retrieval obligations from query-only input."""
    logits = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            4.0,
            0.0,
            0.0,
            0.0,
            0.0,
            -3.0,
            -3.0,
            3.0,
            -3.0,
            -3.0,
            3.0,
            -3.0,
            -3.0,
            -3.0,
            -3.0,
        ],
        dtype=np.float32,
    )

    decision = _v2_query_decision_from_logits(logits)

    assert isinstance(decision, QueryDecision)
    assert decision.query_contract.final_label == ""
    assert decision.answerability_shape.final_label == ""
    assert decision.retrieval_modality.final_label == ""
    assert decision.retrieval_obligation.final_label == ""
    assert decision.retrieval_intents is None
    assert decision.evidence_kinds is None
    assert decision.heads == {}


def test_load_trustworthy_threshold_reads_manifest(tmp_path):
    """Packaged release thresholds should override the default constant."""
    (tmp_path / "manifest.json").write_text(
        '{"release": {"trustworthy_threshold": 0.44}}',
        encoding="utf-8",
    )

    assert _load_trustworthy_threshold(tmp_path) == 0.44


def test_preferred_onnx_path_prefers_fp32_graph(tmp_path):
    """v2 packages should prefer the FP32 ONNX graph for governance accuracy."""
    fp32 = tmp_path / "model.onnx"
    quantized = tmp_path / "model_quantized.onnx"
    fp32.write_bytes(b"fp32")
    quantized.write_bytes(b"int8")

    assert _preferred_onnx_path(tmp_path) == fp32


def test_run_v2_texts_uses_onnx_numpy_inputs():
    """ONNX v2 inference should feed only graph-declared NumPy inputs."""

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
    pyrrho._runtime = "onnx"
    pyrrho._tokenizer = _Tokenizer()
    pyrrho._model = _Session()

    logits = pyrrho._run_v2_texts(["Question: q\n\nSources:\na"])

    assert logits.shape == (1, 18)
    assert set(pyrrho._model.feed) == {"input_ids", "attention_mask"}


def test_required_label_map_rejects_legacy_packages():
    """Packages without the required g5 heads are not supported."""
    try:
        _required_label_map({"retrieval_action_id2label": None}, "retrieval_action_id2label")
    except ValueError as exc:
        assert "Pyrrho g5 package is required" in str(exc)
    else:
        raise AssertionError("Expected stale Pyrrho package rejection.")

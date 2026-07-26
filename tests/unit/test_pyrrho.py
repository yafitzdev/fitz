# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho v2 governance backend."""

from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

import fitz_sage.governance.pyrrho as pyrrho_module
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import (
    COMPROMISED_MODEL_OPT_IN,
    DEFAULT_ONNX_MODEL_FILENAME,
    MODEL_ID,
    PYRRHO_POST_TAG,
    PYRRHO_PRE_TAG,
    V2_MAX_LENGTH,
    GovernanceDecision,
    MultiLabelDecision,
    Pyrrho,
    PyrrhoQueryPlan,
    _format_input,
    _format_query_input,
    _is_v2_model_dir,
    _load_max_length,
    _load_sufficient_threshold,
    _pinned_remote_model,
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
    pyrrho._run_onnx_texts_with_stats = MagicMock(
        return_value=(
            np.asarray([logits]),
            [SimpleNamespace(input_tokens=12, input_truncated=False, max_input_tokens=2048)],
        )
    )

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
    assert plan.input_tokens == 12
    assert plan.input_truncated is False
    pyrrho._run_onnx_texts_with_stats.assert_called_once_with(
        [f"{PYRRHO_PRE_TAG}\nQuestion: Compare release notes and rollout table"]
    )


def test_v2_threshold_and_ontology_fall_back_from_weak_sufficient():
    """A threshold-demoted verdict cannot retain an incompatible `none` failure."""
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
    assert decision.evidence_verdict.final_label == "INSUFFICIENT"
    assert decision.evidence_verdict.used_threshold_fallback is True
    assert decision.evidence_verdict.used_consistency_fallback is True
    assert decision.failure_mode is not None
    assert decision.failure_mode.raw_label == "none"
    assert decision.failure_mode.final_label == "missing_or_incomplete_evidence"
    assert decision.used_consistency_fallback is True
    assert decision.pre_consistency_pair == ("DISPUTED", "none")
    assert decision.mode is AnswerMode.INSUFFICIENT


@pytest.mark.parametrize(
    ("verdict_index", "failure_index", "expected_verdict", "expected_failure"),
    [
        (2, 1, "DISPUTED", "unresolved_conflict"),
        (2, 2, "INSUFFICIENT", "missing_or_incomplete_evidence"),
        (1, 3, "INSUFFICIENT", "wrong_scope_or_version"),
        (0, 0, "INSUFFICIENT", "missing_or_incomplete_evidence"),
        (0, 1, "INSUFFICIENT", "missing_or_incomplete_evidence"),
    ],
)
def test_v2_incompatible_head_pairs_reconcile_fail_closed(
    verdict_index, failure_index, expected_verdict, expected_failure
):
    """Independent head contradictions never upgrade the governance verdict."""
    logits = np.full(18, -4.0, dtype=np.float32)
    logits[verdict_index] = 4.0
    logits[3 + failure_index] = 4.0

    decision = _v2_decision_from_logits(logits)

    assert decision.evidence_verdict is not None
    assert decision.failure_mode is not None
    assert decision.evidence_verdict.final_label == expected_verdict
    assert decision.failure_mode.final_label == expected_failure
    assert decision.used_consistency_fallback is True
    assert "consistency fallback" in decision.reason


def test_load_sufficient_threshold_reads_manifest(tmp_path):
    """Packaged release thresholds should override the default constant."""
    (tmp_path / "manifest.json").write_text(
        '{"release": {"sufficient_threshold": 0.44}}',
        encoding="utf-8",
    )

    assert _load_sufficient_threshold(tmp_path) == 0.44


@pytest.mark.parametrize("value", [0.2, 1.1, "NaN", "not-a-number", True])
def test_load_sufficient_threshold_rejects_unsafe_values(tmp_path, value):
    """Malformed release thresholds must not silently fall back to a default."""
    (tmp_path / "manifest.json").write_text(
        json.dumps({"release": {"sufficient_threshold": value}}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sufficient_threshold"):
        _load_sufficient_threshold(tmp_path)


def test_load_max_length_rejects_encoder_overflow(tmp_path):
    """Release token budgets cannot exceed the encoder position limit."""
    (tmp_path / "manifest.json").write_text('{"release": {"max_length": 9000}}', encoding="utf-8")
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}', encoding="utf-8")

    with pytest.raises(ValueError, match="exceeds encoder limit"):
        _load_max_length(tmp_path)


def test_load_max_length_defaults_to_v2_budget(tmp_path):
    """Packages without release metadata use the current v2 inference budget."""
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}', encoding="utf-8")

    assert V2_MAX_LENGTH == 4096
    assert _load_max_length(tmp_path) == 4096


@pytest.mark.parametrize("value", [2048.5, "2048", True, 0])
def test_load_max_length_rejects_non_integer_values(tmp_path, value):
    (tmp_path / "manifest.json").write_text(
        json.dumps({"release": {"max_length": value}}), encoding="utf-8"
    )
    (tmp_path / "config.json").write_text('{"max_position_embeddings": 8192}', encoding="utf-8")

    with pytest.raises(ValueError, match="positive integer"):
        _load_max_length(tmp_path)


def test_preferred_onnx_path_prefers_fp32_graph(tmp_path):
    """v2 packages should prefer the FP32 ONNX graph for governance accuracy."""
    fp32 = tmp_path / "model.onnx"
    quantized = tmp_path / "model_quantized.onnx"
    fp32.write_bytes(b"fp32")
    quantized.write_bytes(b"int8")

    assert _preferred_onnx_path(tmp_path) == fp32
    assert DEFAULT_ONNX_MODEL_FILENAME == "model.onnx"


def test_preferred_onnx_path_honors_manifest_declared_int8(tmp_path):
    """INT8 is selected only when the package explicitly declares it."""
    fp32 = tmp_path / "model.onnx"
    quantized = tmp_path / "model_quantized.onnx"
    fp32.write_bytes(b"fp32")
    quantized.write_bytes(b"int8")
    report = {
        "passed": True,
        "comparisons": {"native_vs_int8": {"passed": True}},
        "artifacts": {"onnx_int8_sha256": hashlib.sha256(quantized.read_bytes()).hexdigest()},
    }
    report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
    (tmp_path / "onnx_parity_report.json").write_bytes(report_bytes)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "release": {"preferred_onnx_graph": "model_quantized.onnx"},
                "onnx_parity": {
                    "passed": True,
                    "report": "onnx_parity_report.json",
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )

    assert _preferred_onnx_path(tmp_path) == quantized


def test_preferred_onnx_path_rejects_tampered_parity_bound_graph(tmp_path):
    """A graph replaced after parity approval must not load."""
    fp32 = tmp_path / "model.onnx"
    fp32.write_bytes(b"approved")
    report = {
        "passed": True,
        "comparisons": {"native_vs_fp32": {"passed": True}},
        "artifacts": {"onnx_fp32_sha256": hashlib.sha256(b"approved").hexdigest()},
    }
    report_bytes = (json.dumps(report, sort_keys=True) + "\n").encode()
    (tmp_path / "onnx_parity_report.json").write_bytes(report_bytes)
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "release": {"preferred_onnx_graph": "model.onnx"},
                "onnx_parity": {
                    "passed": True,
                    "report": "onnx_parity_report.json",
                    "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    fp32.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="graph hash"):
        _preferred_onnx_path(tmp_path)


def test_preferred_onnx_path_rejects_missing_declared_graph(tmp_path):
    """A stale graph declaration must fail instead of silently changing precision."""
    (tmp_path / "model.onnx").write_bytes(b"fp32")
    (tmp_path / "manifest.json").write_text(
        '{"release": {"preferred_onnx_graph": "model_quantized.onnx"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="declared.*missing"):
        _preferred_onnx_path(tmp_path)


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
            if kwargs.get("return_tensors") is None:
                return {"input_ids": [[1, 2] for _ in texts]}
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


def test_run_onnx_texts_reports_right_truncation():
    """Every governed prefix should expose whether its tail was truncated."""

    class _Input:
        name = "input_ids"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def run(self, output_names, feed):
            return [np.asarray([[0.0] * 18], dtype=np.float32)]

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_tensors") is None:
                return {"input_ids": [list(range(2050))]}
            return {"input_ids": np.asarray([list(range(2048))], dtype=np.int64)}

    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._tokenizer = _Tokenizer()
    pyrrho._model = _Session()
    pyrrho._max_length = 2048

    logits, stats = pyrrho._run_onnx_texts_with_stats(["long evidence"])

    assert logits.shape == (1, 18)
    assert stats[0].input_tokens == 2050
    assert stats[0].max_input_tokens == 2048
    assert stats[0].input_truncated is True


def test_run_onnx_texts_rejects_wrong_logit_width():
    """A reordered/replaced classifier cannot hide behind a runnable ONNX graph."""

    class _Input:
        name = "input_ids"

    class _Session:
        def get_inputs(self):
            return [_Input()]

        def run(self, output_names, feed):
            return [np.asarray([[0.0] * 17], dtype=np.float32)]

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_tensors") is None:
                return {"input_ids": [[1, 2]]}
            return {"input_ids": np.asarray([[1, 2]], dtype=np.int64)}

    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._tokenizer = _Tokenizer()
    pyrrho._model = _Session()

    with pytest.raises(RuntimeError, match="18"):
        pyrrho._run_onnx_texts(["wrong head"])


def test_run_onnx_texts_rejects_missing_declared_input():
    class _Input:
        def __init__(self, name):
            self.name = name

    class _Session:
        def get_inputs(self):
            return [_Input("input_ids"), _Input("attention_mask")]

    class _Tokenizer:
        def __call__(self, texts, **kwargs):
            if kwargs.get("return_tensors") is None:
                return {"input_ids": [[1, 2]]}
            return {"input_ids": np.asarray([[1, 2]], dtype=np.int64)}

    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._tokenizer = _Tokenizer()
    pyrrho._model = _Session()

    with pytest.raises(RuntimeError, match="attention_mask"):
        pyrrho._run_onnx_texts(["missing mask"])


def test_load_does_not_commit_partially_valid_state(tmp_path, monkeypatch):
    """A metadata failure cannot be bypassed by a second load attempt."""
    graph = tmp_path / "model.onnx"
    graph.write_bytes(b"graph")
    monkeypatch.setattr(pyrrho_module, "_resolve_model_dir", lambda model_id: tmp_path)
    monkeypatch.setattr(pyrrho_module, "_is_v2_model_dir", lambda model_dir: True)
    monkeypatch.setattr(pyrrho_module, "_preferred_onnx_path", lambda model_dir: graph)

    def _reject_threshold(model_dir):
        raise ValueError("bad threshold")

    monkeypatch.setattr(pyrrho_module, "_load_sufficient_threshold", _reject_threshold)
    pyrrho = Pyrrho(str(tmp_path))

    with pytest.raises(ValueError, match="bad threshold"):
        pyrrho._load()

    assert pyrrho._tokenizer is None
    assert pyrrho._model is None
    assert pyrrho._model_dir is None
    assert pyrrho._onnx_path is None


def test_load_requires_v2_package_with_onnx(tmp_path):
    """Pyrrho no longer loads packages without the native v2 ONNX contract."""
    (tmp_path / "config.json").write_text('{"id2label": {"0": "OTHER"}}', encoding="utf-8")
    pyrrho = Pyrrho(model_id=str(tmp_path))

    with pytest.raises(ValueError, match="v2 package"):
        pyrrho._load()


def test_is_v2_model_dir_recognizes_native_labels(tmp_path):
    """The package validator should recognize the native v2 label layout."""
    labels = dict(
        enumerate(
            (
                "evidence_verdict.INSUFFICIENT",
                "evidence_verdict.DISPUTED",
                "evidence_verdict.SUFFICIENT",
                "failure_mode.none",
                "failure_mode.unresolved_conflict",
                "failure_mode.missing_or_incomplete_evidence",
                "failure_mode.wrong_scope_or_version",
                "failure_mode.ambiguous_request",
                "retrieval_intents.needs_lookup",
                "retrieval_intents.needs_temporal_resolution",
                "retrieval_intents.needs_comparison_or_set",
                "retrieval_intents.needs_broad_coverage",
                "evidence_kinds.needs_text",
                "evidence_kinds.needs_table_or_record",
                "evidence_kinds.needs_code_or_symbol",
                "evidence_kinds.needs_config_or_setting",
                "evidence_kinds.needs_log_or_run_result",
                "evidence_kinds.needs_document_layout",
            )
        )
    )
    label2id = {label: index for index, label in labels.items()}
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "id2label": labels,
                "label2id": label2id,
                "problem_type": "multi_label_classification",
            }
        ),
        encoding="utf-8",
    )

    assert _is_v2_model_dir(tmp_path) is True


def test_is_v2_model_dir_rejects_reordered_complete_labels(tmp_path):
    """Decoder positions are part of the package ABI, not merely a label set."""
    labels = {
        str(index): label
        for index, label in enumerate(
            [
                "evidence_verdict.DISPUTED",
                "evidence_verdict.INSUFFICIENT",
                "evidence_verdict.SUFFICIENT",
                "failure_mode.none",
                "failure_mode.unresolved_conflict",
                "failure_mode.missing_or_incomplete_evidence",
                "failure_mode.wrong_scope_or_version",
                "failure_mode.ambiguous_request",
                "retrieval_intents.needs_lookup",
                "retrieval_intents.needs_temporal_resolution",
                "retrieval_intents.needs_comparison_or_set",
                "retrieval_intents.needs_broad_coverage",
                "evidence_kinds.needs_text",
                "evidence_kinds.needs_table_or_record",
                "evidence_kinds.needs_code_or_symbol",
                "evidence_kinds.needs_config_or_setting",
                "evidence_kinds.needs_log_or_run_result",
                "evidence_kinds.needs_document_layout",
            ]
        )
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "id2label": labels,
                "label2id": {label: int(index) for index, label in labels.items()},
                "problem_type": "multi_label_classification",
            }
        ),
        encoding="utf-8",
    )

    assert _is_v2_model_dir(tmp_path) is False


def test_default_remote_model_is_quarantined_without_explicit_opt_in(monkeypatch):
    """The known-compromised default must not load during normal operation."""
    monkeypatch.delenv(COMPROMISED_MODEL_OPT_IN, raising=False)

    with pytest.raises(RuntimeError, match="quarantined"):
        _pinned_remote_model(MODEL_ID)


def test_compromised_remote_opt_in_is_pinned_for_forensic_reproduction(monkeypatch):
    """The escape hatch reproduces one immutable historical artifact only."""
    monkeypatch.setenv(COMPROMISED_MODEL_OPT_IN, "1")

    repo_id, revision = _pinned_remote_model(MODEL_ID)

    assert repo_id == MODEL_ID
    assert len(revision) == 40


def test_custom_remote_model_requires_immutable_commit():
    """Future remote packages cannot float with a mutable main branch."""
    with pytest.raises(ValueError, match="owner/repo@commit"):
        _pinned_remote_model("owner/repo")

    revision = "a" * 40
    assert _pinned_remote_model(f"owner/repo@{revision}") == ("owner/repo", revision)

"""Tests for Fitz-Sage's managed Pyrrho ONNX adapter."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fitz_sage.llm.providers.onnx_pyrrho import (
    DEFAULT_MAX_INPUT_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_MODEL_REVISION,
    OnnxPyrrho,
    _pinned_remote_model,
    _validate_model_artifact,
    decision_from_logits,
    empty_evidence_decision,
    normalize_evidence,
    query_plan_from_logits,
)
from fitz_sage.llm.providers.pyrrho_schema import (
    NUM_PYRRHO_LABELS,
    pyrrho_label_names,
)
from fitz_sage.llm.providers.pyrrho_types import (
    GovernanceDecision,
    PyrrhoModelIdentity,
)


def _logits(
    *,
    verdict: tuple[float, float, float] = (0.0, 0.0, 5.0),
    failure: tuple[float, float, float, float, float] = (5.0, 0.0, 0.0, 0.0, 0.0),
) -> list[float]:
    return [
        *verdict,
        *failure,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
        -1.0,
    ]


def _write_model_artifact(path: Path, *, max_length: int = 2048) -> None:
    names = pyrrho_label_names()
    config = {
        "id2label": {str(index): name for index, name in enumerate(names)},
        "label2id": {name: index for index, name in enumerate(names)},
        "problem_type": "multi_label_classification",
        "max_position_embeddings": 8192,
    }
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")
    (path / "model.onnx").write_bytes(b"fake-onnx")
    (path / "manifest.json").write_text(
        json.dumps({"release": {"max_length": max_length}}),
        encoding="utf-8",
    )


def test_empty_evidence_does_not_load_the_model(monkeypatch: pytest.MonkeyPatch) -> None:
    model = OnnxPyrrho("does-not-exist")

    def fail_load() -> None:
        raise AssertionError("empty evidence must not load a model")

    monkeypatch.setattr(model, "_load", fail_load)
    decision = model.decide("question", [])

    assert isinstance(decision, GovernanceDecision)
    assert decision.verdict == "INSUFFICIENT"
    assert decision.failure_mode.final_label == "missing_or_incomplete_evidence"
    assert decision.deterministic is True


def test_empty_evidence_decision_is_serializable() -> None:
    payload = empty_evidence_decision().to_dict()

    assert payload["verdict"] == "INSUFFICIENT"
    assert payload["probabilities"]["INSUFFICIENT"] == 1.0
    assert payload["heads"]["failure_mode"]["final_label"] == ("missing_or_incomplete_evidence")
    assert payload["deterministic"] is True


def test_normalize_evidence_preserves_text_and_source_identity() -> None:
    evidence = normalize_evidence(
        [
            "ATX-123 is not ATX_123",
            {"source_id": "report-7", "content": "AX 156"},
            SimpleNamespace(document_id="log-2", excerpt="AX_156"),
        ]
    )

    assert evidence == [
        {"source_id": "1", "text": "ATX-123 is not ATX_123"},
        {"source_id": "report-7", "text": "AX 156"},
        {"source_id": "log-2", "text": "AX_156"},
    ]


def test_decision_from_logits_preserves_model_output_metadata() -> None:
    decision = decision_from_logits(
        _logits(),
        input_tokens=1536,
        input_truncated=False,
        max_input_tokens=2048,
        model={"graph_sha256": "abc"},
    )

    assert decision.verdict == "SUFFICIENT"
    assert decision.failure_mode.final_label == "none"
    assert decision.input_tokens == 1536
    assert decision.model == {"graph_sha256": "abc"}


def test_decoder_applies_the_accepted_sufficient_threshold() -> None:
    decision = decision_from_logits(
        _logits(
            verdict=(0.0, 0.0, 0.01),
            failure=(0.0, 0.0, 5.0, 0.0, 0.0),
        )
    )

    assert decision.evidence_verdict.raw_label == "SUFFICIENT"
    assert decision.evidence_verdict.threshold_applied is True
    assert decision.verdict == "INSUFFICIENT"


def test_decoder_reconciles_contradictory_model_heads() -> None:
    decision = decision_from_logits(
        _logits(
            verdict=(0.0, 0.0, 5.0),
            failure=(0.0, 0.0, 5.0, 0.0, 0.0),
        )
    )

    assert decision.pre_consistency_pair == (
        "SUFFICIENT",
        "missing_or_incomplete_evidence",
    )
    assert decision.verdict == "INSUFFICIENT"
    assert decision.consistency_applied is True


def test_query_plan_decoder_only_exposes_query_shape_heads() -> None:
    plan = query_plan_from_logits(_logits(), input_tokens=42)

    assert plan.input_tokens == 42
    assert plan.retrieval_intents.final_labels == ("needs_lookup",)
    assert plan.evidence_kinds.final_labels == ("needs_text",)
    assert set(plan.to_dict()) == {
        "retrieval_intents",
        "evidence_kinds",
        "input_tokens",
        "input_truncated",
        "max_input_tokens",
    }


def test_model_reports_token_truncation_and_exact_identity() -> None:
    class FakeTokenizer:
        def __call__(self, texts, *, truncation, padding, **kwargs):
            if not truncation:
                return {"input_ids": [list(range(2050)) for _ in texts]}
            return {
                "input_ids": np.zeros((len(texts), 2048), dtype=np.int64),
                "attention_mask": np.ones((len(texts), 2048), dtype=np.int64),
            }

    class FakeSession:
        def get_inputs(self):
            return [
                SimpleNamespace(name="input_ids"),
                SimpleNamespace(name="attention_mask"),
            ]

        def run(self, _outputs, feed):
            return [np.asarray([_logits() for _ in range(len(feed["input_ids"]))])]

    model = OnnxPyrrho("local-model")
    model._tokenizer = FakeTokenizer()
    model._session = FakeSession()
    model._identity = PyrrhoModelIdentity(
        model_spec="local-model",
        model_directory="C:/model",
        graph="model.onnx",
        graph_sha256="abc",
        max_input_tokens=2048,
        sufficient_threshold=0.34,
    )

    decision = model.decide("question", ["evidence"])

    assert decision.verdict == "SUFFICIENT"
    assert decision.input_tokens == 2050
    assert decision.input_truncated is True
    assert decision.max_input_tokens == DEFAULT_MAX_INPUT_TOKENS
    assert decision.model["graph_sha256"] == "abc"


def test_model_artifact_requires_exact_pyrrho_schema(tmp_path: Path) -> None:
    _write_model_artifact(tmp_path)
    config_path = tmp_path / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["id2label"]["0"] = "wrong"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="exact v2 18-label"):
        _validate_model_artifact(tmp_path)


def test_model_artifact_defaults_to_current_token_contract(tmp_path: Path) -> None:
    _write_model_artifact(tmp_path)

    artifact = _validate_model_artifact(tmp_path)

    assert artifact.max_input_tokens == 2048
    assert artifact.sufficient_threshold == 0.34
    assert artifact.onnx_path.name == "model.onnx"


def test_model_artifact_rejects_length_above_encoder_limit(tmp_path: Path) -> None:
    _write_model_artifact(tmp_path, max_length=9000)

    with pytest.raises(ValueError, match="exceeds encoder limit"):
        _validate_model_artifact(tmp_path)


def test_remote_models_require_an_immutable_revision() -> None:
    with pytest.raises(ValueError, match="must pin"):
        _pinned_remote_model("owner/model")

    revision = "a" * 40
    assert _pinned_remote_model(f"owner/model@{revision}") == ("owner/model", revision)


def test_default_model_resolves_to_the_accepted_revision() -> None:
    assert _pinned_remote_model(DEFAULT_MODEL_ID) == (
        DEFAULT_MODEL_ID,
        DEFAULT_MODEL_REVISION,
    )


def test_decoder_rejects_wrong_logit_count() -> None:
    with pytest.raises(ValueError, match=str(NUM_PYRRHO_LABELS)):
        decision_from_logits([0.0] * (NUM_PYRRHO_LABELS - 1))

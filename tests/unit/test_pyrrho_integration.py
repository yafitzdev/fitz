from __future__ import annotations

from types import SimpleNamespace

import pytest

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.integrations import pyrrho as integration
from fitz_sage.llm.providers.onnx_pyrrho import (
    decision_from_logits,
    empty_evidence_decision,
)


def _result(content: str, source_id: str = "doc-1") -> ReadResult:
    return ReadResult(
        address=Address(
            kind=AddressKind.SECTION,
            source_id=source_id,
            location="section",
            summary="",
            score=0.8,
            metadata={},
        ),
        content=content,
        file_path=f"{source_id}.md",
    )


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("INSUFFICIENT", AnswerMode.INSUFFICIENT),
        ("DISPUTED", AnswerMode.DISPUTED),
        ("SUFFICIENT", AnswerMode.SUFFICIENT),
    ],
)
def test_verdict_mapping_is_mechanical(verdict: str, expected: AnswerMode) -> None:
    assert integration.answer_mode_from_pyrrho(SimpleNamespace(verdict=verdict)) is expected


def test_unknown_verdict_is_an_error() -> None:
    with pytest.raises(ValueError, match="unknown verdict"):
        integration.answer_mode_from_pyrrho(SimpleNamespace(verdict="MAYBE"))


def test_evidence_is_passed_without_identifier_or_content_cleanup() -> None:
    results = [
        _result("ATX-123", "ATX-123"),
        _result("ATX_123", "ATX_123"),
    ]

    assert integration.pyrrho_evidence(results) == [
        {"source_id": "ATX-123", "text": "ATX-123"},
        {"source_id": "ATX_123", "text": "ATX_123"},
    ]


def test_decide_returns_the_exact_pyrrho_object() -> None:
    expected = empty_evidence_decision()

    class Runtime:
        def decide(self, query, evidence):
            assert query == "question"
            assert evidence == [{"source_id": "doc-1", "text": "evidence"}]
            return expected

    assert integration.decide(Runtime(), "question", [_result("evidence")]) is expected
    assert integration.decision_payload(expected) == expected.to_dict()


def test_pyrrho_failure_propagates_instead_of_becoming_insufficient() -> None:
    class Runtime:
        def decide(self, query, evidence):
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        integration.decide(Runtime(), "question", [])


def test_decision_payload_preserves_probabilities_heads_and_model_identity() -> None:
    logits = [
        0.0,
        0.0,
        5.0,
        5.0,
        0.0,
        0.0,
        0.0,
        0.0,
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
    decision = decision_from_logits(logits, model={"graph_sha256": "abc"})

    assert integration.decision_payload(decision) == decision.to_dict()
    assert integration.decision_payload(decision)["model"]["graph_sha256"] == "abc"


def test_create_pyrrho_only_accepts_pyrrho_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[str | None] = []

    class FakePyrrho:
        def __init__(self, model_spec=None):
            created.append(model_spec)

    monkeypatch.setattr(integration, "OnnxPyrrho", FakePyrrho)

    integration.create_pyrrho("pyrrho/C:/models/release")
    integration.create_pyrrho("pyrrho")
    assert created == ["C:/models/release", None]

    with pytest.raises(ValueError, match="Unknown governance provider"):
        integration.create_pyrrho("other/model")

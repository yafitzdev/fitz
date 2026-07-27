# tests/unit/test_fitz_service_evidence.py
from unittest.mock import Mock, patch

from fitz_sage.api.models.schemas import EvidenceResponse
from fitz_sage.core import EvidenceItem, EvidencePack
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.services.fitz_service import FitzService


def test_service_exposes_answer_without_query_alias() -> None:
    assert hasattr(FitzService, "answer")
    assert not hasattr(FitzService, "query")


def test_service_evidence_returns_engine_evidence_pack() -> None:
    pack = EvidencePack(
        query="What is indexed?",
        mode=AnswerMode.SUFFICIENT,
        items=[
            EvidenceItem(
                rank=1,
                source_id="doc-1",
                file_path="docs/example.md",
                address_kind="section",
                address_location="Overview",
                line_range=(1, 3),
                score=0.91,
                excerpt="Indexed content",
                content="Indexed content with context",
                metadata={"kind": "section"},
            )
        ],
        reasons=["Pyrrho: sources support a confident answer."],
        timings={"retrieval": 0.01},
        indexing_status={"complete": True},
        metadata={"evidence_delivery": {"selected": 1}},
    )
    engine = Mock()
    engine.evidence.return_value = pack

    with patch("fitz_sage.runtime.create_engine", return_value=engine):
        result = FitzService().evidence("What is indexed?", collection="docs")

    engine.load.assert_called_once_with("docs")
    engine.evidence.assert_called_once()
    query = engine.evidence.call_args.args[0]
    assert query.text == "What is indexed?"
    assert result is pack


def test_evidence_response_accepts_pack_dict() -> None:
    pack = EvidencePack(
        query="What is indexed?",
        mode=AnswerMode.INSUFFICIENT,
        items=[],
        reasons=["No relevant evidence retrieved."],
        timings={},
        indexing_status={"complete": False},
        metadata={},
    )

    response = EvidenceResponse(**pack.to_dict())

    assert response.query == "What is indexed?"
    assert response.mode == "insufficient"
    assert response.reasons == ["No relevant evidence retrieved."]
    assert response.indexing_status == {"complete": False}


def test_service_reuses_collection_bound_engine() -> None:
    pack = EvidencePack(query="question", mode=AnswerMode.SUFFICIENT)
    engine = Mock()
    engine.evidence.return_value = pack
    service = FitzService()

    with patch("fitz_sage.runtime.create_engine", return_value=engine) as create:
        service.evidence("first", collection="docs")
        service.evidence("second", collection="docs")

    create.assert_called_once_with("fitz_krag")
    engine.load.assert_called_once_with("docs")
    assert engine.evidence.call_count == 2


def test_service_trace_returns_the_engine_execution_record() -> None:
    run = Mock()
    engine = Mock()
    engine.trace.return_value = run

    with patch("fitz_sage.runtime.create_engine", return_value=engine):
        result = FitzService().trace("What is indexed?", collection="docs")

    engine.load.assert_called_once_with("docs")
    engine.trace.assert_called_once()
    assert engine.trace.call_args.args[0].text == "What is indexed?"
    assert result is run

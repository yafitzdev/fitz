# tests/unit/test_fitz_service_evidence.py
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from fitz_sage.api.models.schemas import EvidenceResponse
from fitz_sage.core import EvidenceItem, EvidencePack
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.core.paths import FitzPaths
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


def test_service_runs_cached_engine_queries_concurrently() -> None:
    pack = EvidencePack(query="question", mode=AnswerMode.SUFFICIENT)
    engine = Mock()
    active = 0
    maximum_active = 0
    state_lock = threading.Lock()
    both_active = threading.Event()

    def evidence(_query):
        nonlocal active, maximum_active
        with state_lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                both_active.set()
        try:
            assert both_active.wait(timeout=2.0)
            return pack
        finally:
            with state_lock:
                active -= 1

    engine.evidence.side_effect = evidence
    service = FitzService()

    with (
        patch("fitz_sage.runtime.create_engine", return_value=engine) as create,
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        futures = [
            executor.submit(service.evidence, question, collection="docs")
            for question in ("first", "second")
        ]
        assert [future.result(timeout=3.0) for future in futures] == [pack, pack]

    assert maximum_active == 2
    create.assert_called_once_with("fitz_krag")
    engine.load.assert_called_once_with("docs")


def test_collection_deletion_waits_for_active_queries(tmp_path) -> None:
    pack = EvidencePack(query="question", mode=AnswerMode.SUFFICIENT)
    engine = Mock()
    query_started = threading.Event()
    release_query = threading.Event()

    def evidence(_query):
        query_started.set()
        assert release_query.wait(timeout=2.0)
        return pack

    engine.evidence.side_effect = evidence
    connection_manager = Mock()
    connection_manager.delete_collection.return_value = True
    service = FitzService()

    FitzPaths.set_workspace(tmp_path / ".fitz")
    try:
        with (
            patch("fitz_sage.runtime.create_engine", return_value=engine),
            patch.object(service, "_connection_manager", return_value=connection_manager),
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            query = executor.submit(service.evidence, "question", collection="docs")
            assert query_started.wait(timeout=1.0)
            slot = next(iter(service._engines.values()))
            deletion = executor.submit(service.delete_collection, "docs")

            with slot._condition:
                assert slot._condition.wait_for(lambda: slot._closing, timeout=1.0)

            assert not deletion.done()
            connection_manager.delete_collection.assert_not_called()

            release_query.set()
            assert query.result(timeout=2.0) is pack
            assert deletion.result(timeout=2.0) is True
    finally:
        FitzPaths.reset()

    engine.stop_background_enrichment.assert_called_once_with()
    connection_manager.delete_collection.assert_called_once_with("docs")


def test_collection_deletion_removes_persisted_manifest_state(tmp_path) -> None:
    workspace = tmp_path / ".fitz"
    collection_dir = workspace / "collections" / "docs"
    collection_dir.mkdir(parents=True)
    (collection_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (collection_dir / "source_dir.txt").write_text("docs", encoding="utf-8")

    connection_manager = Mock()
    connection_manager.delete_collection.return_value = False
    service = FitzService()

    FitzPaths.set_workspace(workspace)
    try:
        with patch.object(service, "_connection_manager", return_value=connection_manager):
            assert service.delete_collection("docs") is True
    finally:
        FitzPaths.reset()

    assert not collection_dir.exists()
    connection_manager.delete_collection.assert_called_once_with("docs")


def test_point_waits_for_active_queries(tmp_path) -> None:
    pack = EvidencePack(query="question", mode=AnswerMode.SUFFICIENT)
    manifest = Mock()
    engine = Mock()
    query_started = threading.Event()
    release_query = threading.Event()
    point_started = threading.Event()

    def evidence(_query):
        query_started.set()
        assert release_query.wait(timeout=2.0)
        return pack

    def point(*_args, **_kwargs):
        point_started.set()
        return manifest

    engine.evidence.side_effect = evidence
    engine.point.side_effect = point
    source = tmp_path / "docs"
    source.mkdir()
    service = FitzService()

    with (
        patch("fitz_sage.runtime.create_engine", return_value=engine),
        ThreadPoolExecutor(max_workers=2) as executor,
    ):
        query = executor.submit(service.evidence, "question", collection="docs")
        assert query_started.wait(timeout=1.0)
        slot = next(iter(service._engines.values()))
        indexing = executor.submit(service.point, source, collection="docs")

        with slot._condition:
            assert slot._condition.wait_for(lambda: slot._waiting_writers == 1, timeout=1.0)

        assert not point_started.is_set()
        release_query.set()
        assert query.result(timeout=2.0) is pack
        assert indexing.result(timeout=2.0) is manifest

    engine.point.assert_called_once_with(source.resolve(), "docs", start_worker=True)

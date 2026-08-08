from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.progressive.manifest import (
    EnrichmentState,
    FileManifest,
    FileState,
    FinalizationState,
    ManifestEntry,
)
from fitz_sage.engines.fitz_krag.progressive.worker import BackgroundEnrichmentWorker


def _entry(
    rel_path: str,
    *,
    enrichment: EnrichmentState = EnrichmentState.PENDING,
) -> ManifestEntry:
    return ManifestEntry(
        file_id=f"id-{rel_path}",
        rel_path=rel_path,
        abs_path=f"/src/{rel_path}",
        content_hash="hash",
        file_type=Path(rel_path).suffix,
        size_bytes=100,
        state=FileState.INDEXED,
        enrichment_state=enrichment,
    )


def _worker(manifest: FileManifest, core: MagicMock) -> BackgroundEnrichmentWorker:
    return BackgroundEnrichmentWorker(manifest, core, MagicMock())


def test_worker_runs_only_background_enrichment(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md"))
    core = MagicMock()
    worker = _worker(manifest, core)

    worker.run_until_complete()

    core.link_entities_file.assert_called_once_with("id-guide.md", ".md")
    core.build_hierarchy_file.assert_called_once_with("id-guide.md", ".md")
    core.build_corpus_hierarchy.assert_called_once_with()
    assert not hasattr(core, "parse_file") or core.parse_file.call_count == 0
    entry = manifest.get("guide.md")
    assert entry is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.COMPLETE
    assert manifest.finalization_status()[0] == FinalizationState.COMPLETE


def test_synchronous_worker_releases_collection_lock(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    write_lock = MagicMock()
    worker = BackgroundEnrichmentWorker(manifest, MagicMock(), write_lock)

    worker.run_until_complete()

    write_lock.release.assert_called_once_with()


def test_background_worker_holds_lock_until_stopped(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    write_lock = MagicMock()
    worker = BackgroundEnrichmentWorker(manifest, MagicMock(), write_lock)

    worker.start()
    worker.wait()
    write_lock.release.assert_not_called()

    worker.stop()

    write_lock.release.assert_called_once_with()


def test_entity_failure_does_not_remove_source_index(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md"))
    core = MagicMock()
    core.link_entities_file.side_effect = RuntimeError("model unavailable")

    _worker(manifest, core).run_until_complete()

    entry = manifest.get("guide.md")
    assert entry is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.FAILED
    assert entry.enrichment_failure_stage == "entities"


def test_hierarchy_retry_starts_after_entity_linking(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md", enrichment=EnrichmentState.ENTITY_LINKED))
    core = MagicMock()

    _worker(manifest, core).run_until_complete()

    core.link_entities_file.assert_not_called()
    core.build_hierarchy_file.assert_called_once()
    assert manifest.get("guide.md").enrichment_state == EnrichmentState.COMPLETE


def test_collection_failure_is_reported_separately(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md", enrichment=EnrichmentState.COMPLETE))
    core = MagicMock()
    core.build_corpus_hierarchy.side_effect = RuntimeError("summary failed")

    _worker(manifest, core).run_until_complete()

    state, failure = manifest.finalization_status()
    assert state == FinalizationState.FAILED
    assert failure == "summary failed"
    assert manifest.get("guide.md").state == FileState.INDEXED


def test_query_boost_prioritizes_file_and_sibling(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("docs/hot.md"))
    manifest.add(_entry("docs/sibling.md"))
    manifest.add(_entry("other/cold.md"))
    worker = _worker(manifest, MagicMock())

    worker.boost_files(["docs/hot.md"])

    assert manifest.get("docs/hot.md").priority == 1
    assert manifest.get("docs/sibling.md").priority == 2
    assert manifest.get("other/cold.md").priority == 4


def test_worker_stays_paused_until_all_concurrent_queries_finish(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    worker = _worker(manifest, MagicMock())

    worker.signal_query_start()
    worker.signal_query_start()
    worker.signal_query_end()

    assert worker._query_active.is_set()
    assert worker._active_queries == 1

    worker.signal_query_end()

    assert not worker._query_active.is_set()
    assert worker._active_queries == 0


def test_warm_target_requires_completed_enrichment_and_query(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md", enrichment=EnrichmentState.COMPLETE))
    worker = _worker(manifest, MagicMock())
    assert worker._next_warm_target() is None

    manifest.bump_priority(["guide.md"])

    assert worker._next_warm_target().rel_path == "guide.md"


def test_demand_summary_failure_is_recorded_for_retry(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    entry = _entry("guide.md", enrichment=EnrichmentState.COMPLETE)
    manifest.add(entry)
    core = MagicMock()
    core.summarize_file.side_effect = RuntimeError("model unavailable")
    worker = _worker(manifest, core)

    worker._summarize_entry(entry)

    failed = manifest.get("guide.md")
    assert failed is not None
    assert failed.enrichment_state == EnrichmentState.FAILED
    assert failed.enrichment_failure_stage == "summary"
    assert failed.enrichment_failure_message == "model unavailable"

from pathlib import Path

from fitz_sage.engines.fitz_krag.progressive.manifest import (
    EnrichmentState,
    FileManifest,
    FileState,
    FinalizationState,
    ManifestEntry,
    indexing_status,
)


def _entry(
    rel_path: str,
    *,
    state: FileState = FileState.REGISTERED,
    enrichment: EnrichmentState = EnrichmentState.PENDING,
) -> ManifestEntry:
    return ManifestEntry(
        file_id=f"id-{rel_path}",
        rel_path=rel_path,
        abs_path=f"/src/{rel_path}",
        content_hash="hash",
        file_type=Path(rel_path).suffix,
        size_bytes=100,
        state=state,
        enrichment_state=enrichment,
    )


def test_index_and_enrichment_status_are_independent(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(
        _entry(
            "ready.md",
            state=FileState.INDEXED,
            enrichment=EnrichmentState.PENDING,
        )
    )
    manifest.add(_entry("waiting.md"))

    status = indexing_status(manifest)

    assert status["indexed"] == 1
    assert status["pending"] == 1
    assert status["query_ready"] is False
    assert status["enrichment"]["pending"] == 1
    assert status["by_index_state"] == {"indexed": 1, "registered": 1}
    assert status["by_enrichment_state"] == {"pending": 1}


def test_enrichment_failure_keeps_file_searchable(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md", state=FileState.INDEXED))

    manifest.mark_enrichment_failed("guide.md", stage="hierarchy", message="model failed")
    entry = manifest.get("guide.md")
    status = indexing_status(manifest)

    assert entry is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.FAILED
    assert status["complete"] is True
    assert status["enrichment"]["failed"] == 1
    assert status["enrichment"]["complete"] is False


def test_index_failure_is_not_query_pending(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("broken.md"))
    manifest.mark_failed("broken.md", stage="parse", message="bad document")

    status = indexing_status(manifest)

    assert status["pending"] == 0
    assert status["query_ready"] is True
    assert status["complete"] is False
    assert status["failed_files"][0]["stage"] == "parse"


def test_retry_resumes_hierarchy_without_repeating_entity_stage(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(_entry("guide.md", state=FileState.INDEXED))
    manifest.mark_enrichment_failed("guide.md", stage="hierarchy", message="timeout")
    manifest.mark_finalization_failed("timeout")

    manifest.prepare_enrichment_retry()
    entry = manifest.get("guide.md")
    finalization, failure = manifest.finalization_status()

    assert entry is not None
    assert entry.enrichment_state == EnrichmentState.ENTITY_LINKED
    assert finalization == FinalizationState.PENDING
    assert failure is None


def test_retry_resumes_failed_summary_after_completed_hierarchy(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(
        _entry(
            "guide.md",
            state=FileState.INDEXED,
            enrichment=EnrichmentState.COMPLETE,
        )
    )
    manifest.mark_enrichment_failed("guide.md", stage="summary", message="timeout")

    manifest.prepare_enrichment_retry()

    entry = manifest.get("guide.md")
    assert entry is not None
    assert entry.enrichment_state == EnrichmentState.COMPLETE
    assert entry.enrichment_failure_stage is None
    assert entry.enrichment_failure_message is None


def test_manifest_round_trip_persists_both_lifecycles(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    manifest = FileManifest(path)
    manifest.add(
        _entry(
            "guide.md",
            state=FileState.INDEXED,
            enrichment=EnrichmentState.COMPLETE,
        )
    )
    manifest.mark_finalized()
    manifest.save()

    loaded = FileManifest(path)
    entry = loaded.get("guide.md")

    assert entry is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.COMPLETE
    assert loaded.finalization_status() == (FinalizationState.COMPLETE, None)


def test_priority_updates_preserve_lifecycle_state(tmp_path: Path) -> None:
    manifest = FileManifest(tmp_path / "manifest.json")
    manifest.add(
        _entry(
            "guide.md",
            state=FileState.INDEXED,
            enrichment=EnrichmentState.COMPLETE,
        )
    )

    manifest.bump_priority(["guide.md"])
    entry = manifest.get("guide.md")

    assert entry is not None
    assert entry.priority == 1
    assert entry.last_queried_at is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.COMPLETE


def test_none_manifest_reports_ready_empty_collection() -> None:
    status = indexing_status(None)

    assert status["complete"] is True
    assert status["query_ready"] is True
    assert status["enrichment"]["complete"] is True


def test_empty_manifest_has_no_enrichment_work(tmp_path: Path) -> None:
    status = indexing_status(FileManifest(tmp_path / "manifest.json"))

    assert status["query_ready"] is True
    assert status["enrichment"]["finalization"] == "complete"
    assert status["enrichment"]["complete"] is True

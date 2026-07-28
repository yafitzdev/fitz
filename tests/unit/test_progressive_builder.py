from pathlib import Path
from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.progressive.builder import ManifestBuilder
from fitz_sage.engines.fitz_krag.progressive.manifest import (
    EnrichmentState,
    FileState,
)


def _config() -> SimpleNamespace:
    return SimpleNamespace(
        code_languages=["python", "typescript", "java", "go"],
        table_extensions=[".csv", ".tsv"],
    )


def test_builder_discovers_and_hashes_without_parsing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "guide.md").write_text("# Guide\nSearchable text", encoding="utf-8")
    (source / "design.psd").write_bytes(b"unsupported")

    manifest = ManifestBuilder(_config()).build(source, tmp_path / "manifest.json")

    entries = manifest.entries()
    assert entries["guide.md"].state == FileState.REGISTERED
    assert entries["guide.md"].content_hash
    assert entries["guide.md"].enrichment_state == EnrichmentState.PENDING
    assert entries["design.psd"].state == FileState.UNSUPPORTED
    assert entries["design.psd"].enrichment_state == EnrichmentState.NOT_APPLICABLE


def test_unchanged_indexed_file_keeps_enrichment_state(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "guide.md"
    path.write_text("stable", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    builder = ManifestBuilder(_config())
    first = builder.build(source, manifest_path)
    first.update_state("guide.md", FileState.INDEXED)
    first.update_enrichment_state("guide.md", EnrichmentState.COMPLETE)
    first.mark_finalized()
    first.save()

    second = builder.build(source, manifest_path)
    entry = second.get("guide.md")

    assert entry is not None
    assert entry.state == FileState.INDEXED
    assert entry.enrichment_state == EnrichmentState.COMPLETE
    assert second.finalization_status()[0].value == "complete"


def test_changed_file_returns_to_registered_and_resets_finalization(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    path = source / "guide.md"
    path.write_text("v1", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    builder = ManifestBuilder(_config())
    first = builder.build(source, manifest_path)
    first.update_state("guide.md", FileState.INDEXED)
    first.update_enrichment_state("guide.md", EnrichmentState.COMPLETE)
    first.mark_finalized()
    first.save()

    path.write_text("v2", encoding="utf-8")
    second = builder.build(source, manifest_path)
    entry = second.get("guide.md")

    assert entry is not None
    assert entry.state == FileState.REGISTERED
    assert entry.enrichment_state == EnrichmentState.PENDING
    assert second.finalization_status()[0].value == "pending"

# tests/unit/test_progressive_builder.py
"""Tests for progressive manifest building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.progressive.builder import ManifestBuilder
from fitz_sage.engines.fitz_krag.progressive.manifest import FileState


def test_manifest_builder_skips_fitz_workspace(tmp_path: Path) -> None:
    """Internal .fitz workspace files must never become source documents."""
    (tmp_path / "README.md").write_text("# Corpus\nUseful notes.", encoding="utf-8")

    internal_dir = tmp_path / ".fitz" / "collections" / "rag_test_corpus"
    internal_dir.mkdir(parents=True)
    (internal_dir / "source_dir.txt").write_text(str(tmp_path), encoding="utf-8")
    parsed_dir = internal_dir / "parsed"
    parsed_dir.mkdir()
    (parsed_dir / "parsed-cache.txt").write_text("cached parser output", encoding="utf-8")

    builder = ManifestBuilder(MagicMock())
    manifest = builder.build(
        tmp_path,
        tmp_path / ".fitz" / "collections" / "rag_test_corpus" / "manifest.json",
    )

    assert set(manifest.entries()) == {"README.md"}


def test_manifest_builder_uses_the_configured_format_contract(tmp_path: Path) -> None:
    (tmp_path / "policy.yaml").write_text("owner: security", encoding="utf-8")
    (tmp_path / "schema.sql").write_text("CREATE TABLE users(id INT);", encoding="utf-8")
    (tmp_path / "records.tsv").write_text("id\towner\n1\tops\n", encoding="utf-8")
    (tmp_path / "design.psd").write_bytes(b"not a supported document")
    (tmp_path / ".env").write_text("SECRET=not-indexed", encoding="utf-8")

    config = MagicMock()
    config.code_languages = ["python"]
    config.table_extensions = [".csv", ".tsv"]
    manifest = ManifestBuilder(config).build(
        tmp_path,
        tmp_path / ".fitz" / "collections" / "formats" / "manifest.json",
    )

    entries = manifest.entries()
    assert entries["policy.yaml"].state == FileState.REGISTERED
    assert entries["schema.sql"].state == FileState.REGISTERED
    assert entries["records.tsv"].state == FileState.REGISTERED
    assert entries["design.psd"].state == FileState.UNSUPPORTED
    assert entries[".env"].state == FileState.UNSUPPORTED

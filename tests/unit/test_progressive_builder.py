# tests/unit/test_progressive_builder.py
"""Tests for progressive manifest building."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.progressive.builder import ManifestBuilder


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

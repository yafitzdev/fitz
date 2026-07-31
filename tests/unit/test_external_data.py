"""Tests for shared external-corpus safety helpers."""

from __future__ import annotations

import zipfile

from benchmarks.fitz_bench.external_data import safe_extract_zip


def test_non_atomic_extraction_replaces_output_without_directory_rename(tmp_path) -> None:
    archive = tmp_path / "corpus.zip"
    output = tmp_path / "corpus"
    output.mkdir()
    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with zipfile.ZipFile(archive, "w") as target:
        target.writestr("source/document.txt", "content")

    files, extracted_bytes = safe_extract_zip(
        archive,
        output,
        max_extracted_bytes=100,
        atomic=False,
    )

    assert files == 1
    assert extracted_bytes == len("content")
    assert not (output / "stale.txt").exists()
    assert (output / "source/document.txt").read_text(encoding="utf-8") == "content"

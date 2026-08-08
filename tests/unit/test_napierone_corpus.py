"""Tests for safe, reproducible NapierOne corpus preparation."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import pytest

from benchmarks.fitz_bench.external_data import safe_extract_zip
from benchmarks.fitz_bench.napierone import (
    archive_spec,
    parse_sha256_manifest,
    prepare_corpus,
)


def test_archive_spec_uses_published_napierone_layout() -> None:
    spec = archive_spec("pdf", "tiny")

    assert spec.key == "NapierOne/Data/PDF/PDF-tiny.zip"
    assert spec.hash_key == "NapierOne/Data/PDF/PDF-tiny_zip_hashes.txt"


def test_prepare_corpus_reuses_verified_offline_archive(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    archive = archives / "JSON-tiny.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("JSON-tiny/JSON-tiny-0001.json", '{"status": "ok"}')

    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (archives / "JSON-tiny_zip_hashes.txt").write_text(
        f"SHA256             32               {digest}\n",
        encoding="utf-8",
    )

    prepared = prepare_corpus(
        tmp_path,
        file_types=["JSON"],
        profile="tiny",
        offline=True,
        max_extracted_bytes=1024,
    )

    corpus = Path(prepared.corpus_dir)
    assert prepared.archives[0].extracted_files == 1
    assert (corpus / "JSON" / "JSON-tiny" / "JSON-tiny-0001.json").exists()
    assert prepared.file_types == ("JSON",)
    assert prepared.materialization in {"hardlink", "copy"}


def test_prepare_corpus_isolates_each_selected_type(tmp_path) -> None:
    archives = tmp_path / "archives"
    archives.mkdir()
    for file_type, body in (("JSON", "{}"), ("TXT", "plain text")):
        archive = archives / f"{file_type}-tiny.zip"
        with zipfile.ZipFile(archive, "w") as output:
            output.writestr(
                f"{file_type}-tiny/{file_type}-tiny-0001.{file_type.lower()}",
                body,
            )
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (archives / f"{file_type}-tiny_zip_hashes.txt").write_text(
            f"SHA256 32 {digest}\n",
            encoding="utf-8",
        )

    prepare_corpus(
        tmp_path,
        file_types=["JSON"],
        profile="tiny",
        offline=True,
        max_extracted_bytes=1024,
    )
    prepared = prepare_corpus(
        tmp_path,
        file_types=["TXT"],
        profile="tiny",
        offline=True,
        max_extracted_bytes=1024,
    )

    selected = Path(prepared.corpus_dir)
    assert (selected / "TXT").is_dir()
    assert not (selected / "JSON").exists()


def test_safe_extract_rejects_parent_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("../outside.txt", "not allowed")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(
            archive,
            tmp_path / "corpus" / "JSON",
            max_extracted_bytes=1024,
        )

    assert not (tmp_path / "outside.txt").exists()


def test_safe_extract_rejects_windows_parent_traversal(tmp_path) -> None:
    archive = tmp_path / "unsafe-windows.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(r"..\outside.txt", "not allowed")

    with pytest.raises(ValueError, match="Unsafe ZIP member"):
        safe_extract_zip(
            archive,
            tmp_path / "corpus" / "JSON",
            max_extracted_bytes=1024,
        )


def test_parse_sha256_manifest_requires_published_hash() -> None:
    digest = "a" * 64

    assert parse_sha256_manifest(f"SHA256 32 {digest}") == digest
    with pytest.raises(ValueError, match="does not contain"):
        parse_sha256_manifest("MD5 16 deadbeef")

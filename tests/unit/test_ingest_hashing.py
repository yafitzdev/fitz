"""Tests for canonical ingestion content hashing."""

from pathlib import Path

import pytest

from fitz_sage.ingestion.hashing import compute_bytes_hash, compute_content_hash


def test_file_hash_is_stable_sha256(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("identical content", encoding="utf-8")
    second.write_text("identical content", encoding="utf-8")

    digest = compute_content_hash(first)

    assert digest == compute_content_hash(second)
    assert len(digest) == 64
    assert all(character in "0123456789abcdef" for character in digest)


def test_file_hash_distinguishes_content(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("content A", encoding="utf-8")
    second.write_text("content B", encoding="utf-8")

    assert compute_content_hash(first) != compute_content_hash(second)


def test_file_hash_rejects_missing_file_and_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_content_hash(tmp_path / "missing.txt")
    with pytest.raises(IsADirectoryError):
        compute_content_hash(tmp_path)


def test_file_and_bytes_hash_use_same_identity(tmp_path: Path) -> None:
    payload = b"\x00\x01binary payload"
    source = tmp_path / "payload.bin"
    source.write_bytes(payload)

    assert compute_content_hash(source) == compute_bytes_hash(payload)

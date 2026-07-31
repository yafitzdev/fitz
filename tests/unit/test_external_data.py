"""Tests for shared external-corpus safety helpers."""

from __future__ import annotations

from pathlib import Path

from benchmarks.fitz_bench.external_data import _replace_directory_with_retry


def test_directory_replace_retries_transient_windows_handle(
    tmp_path,
    monkeypatch,
) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    source.mkdir()
    (source / "document.txt").write_text("content", encoding="utf-8")
    original = Path.replace
    calls = 0

    def flaky_replace(path: Path, target: Path) -> Path:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise PermissionError("transient scanner handle")
        return original(path, target)

    monkeypatch.setattr(Path, "replace", flaky_replace)
    monkeypatch.setattr("benchmarks.fitz_bench.external_data.time.sleep", lambda _delay: None)

    _replace_directory_with_retry(source, destination, attempts=3)

    assert calls == 3
    assert (destination / "document.txt").read_text(encoding="utf-8") == "content"

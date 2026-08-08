"""Tests for cross-process collection writer ownership."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fitz_sage.engines.fitz_krag.progressive.write_lock import (
    CollectionBusyError,
    CollectionWriteLock,
)

_CHILD_ATTEMPT = """
import sys
from pathlib import Path
from fitz_sage.engines.fitz_krag.progressive.write_lock import (
    CollectionBusyError,
    CollectionWriteLock,
)

lock = CollectionWriteLock(Path(sys.argv[1]), collection="docs", operation="child write")
try:
    lock.acquire()
except CollectionBusyError as exc:
    print(exc)
    raise SystemExit(23)
else:
    lock.release()
"""

_CHILD_HOLD = """
import sys
from pathlib import Path
from fitz_sage.engines.fitz_krag.progressive.write_lock import CollectionWriteLock

lock = CollectionWriteLock(Path(sys.argv[1]), collection="docs", operation="child write")
lock.acquire()
print("acquired", flush=True)
sys.stdin.read()
"""


def _child_attempt(collection_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _CHILD_ATTEMPT, str(collection_dir)],
        capture_output=True,
        check=False,
        text=True,
        timeout=15,
    )


def test_lock_rejects_an_independent_process_until_released(tmp_path: Path) -> None:
    collection_dir = tmp_path / "collections" / "docs"
    lock = CollectionWriteLock(
        collection_dir,
        collection="docs",
        operation="source indexing",
    )
    lock.acquire()
    try:
        blocked = _child_attempt(collection_dir)
    finally:
        lock.release()

    assert blocked.returncode == 23
    assert "Collection 'docs' is busy." in blocked.stdout

    accepted = _child_attempt(collection_dir)
    assert accepted.returncode == 0, accepted.stderr


def test_second_lock_in_same_process_is_rejected(tmp_path: Path) -> None:
    collection_dir = tmp_path / "collections" / "docs"
    first = CollectionWriteLock(
        collection_dir,
        collection="docs",
        operation="source indexing",
    )
    second = CollectionWriteLock(
        collection_dir,
        collection="docs",
        operation="background enrichment",
    )

    with first:
        with pytest.raises(CollectionBusyError, match="Collection 'docs' is busy"):
            second.acquire()

    second.acquire()
    second.release()


def test_process_exit_releases_the_lock_without_deleting_its_file(tmp_path: Path) -> None:
    collection_dir = tmp_path / "collections" / "docs"
    process = subprocess.Popen(
        [sys.executable, "-c", _CHILD_HOLD, str(collection_dir)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "acquired"
        process.terminate()
        process.wait(timeout=15)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=15)

    lock = CollectionWriteLock(
        collection_dir,
        collection="docs",
        operation="source indexing",
    )
    lock.acquire()
    lock.release()

    assert lock.path.exists()


def test_release_is_idempotent(tmp_path: Path) -> None:
    lock = CollectionWriteLock(
        tmp_path,
        collection="docs",
        operation="source indexing",
    )

    lock.acquire()
    lock.release()
    lock.release()

    assert not lock.acquired

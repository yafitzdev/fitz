"""Cross-process ownership for collection indexing and enrichment."""

from __future__ import annotations

import errno
import json
import os
import socket
import sys
import time
from pathlib import Path
from typing import BinaryIO

from fitz_sage.core.exceptions import KnowledgeError


class CollectionBusyError(KnowledgeError):
    """Raised when another process already owns a collection's write path."""

    def __init__(self, message: str, *, owner: dict[str, object] | None = None) -> None:
        super().__init__(message)
        self.owner = dict(owner or {})

    @property
    def operation(self) -> str | None:
        """Return the active writer operation when lock metadata is available."""
        operation = self.owner.get("operation")
        return operation if isinstance(operation, str) else None


class CollectionWriteLock:
    """Own the single-writer path for one collection until explicitly released."""

    def __init__(
        self,
        collection_dir: Path,
        *,
        collection: str,
        operation: str,
    ) -> None:
        self._path = collection_dir / "writer.lock"
        self._collection = collection
        self._operation = operation
        self._handle: BinaryIO | None = None

    @property
    def acquired(self) -> bool:
        """Return whether this object currently owns the OS lock."""
        return self._handle is not None

    @property
    def path(self) -> Path:
        """Return the persistent lock-file path."""
        return self._path

    def acquire(self) -> None:
        """Acquire without waiting, failing clearly when another writer is active."""
        if self._handle is not None:
            raise RuntimeError("Collection write lock is already acquired.")

        self._path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self._path, os.O_RDWR | os.O_CREAT, 0o600)
        handle = os.fdopen(descriptor, "r+b", buffering=0)
        try:
            if os.fstat(descriptor).st_size == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            _acquire_native_lock(handle)
        except OSError as exc:
            owner = _read_owner(handle)
            handle.close()
            if _is_lock_contention(exc):
                raise CollectionBusyError(self._busy_message(owner), owner=owner) from exc
            raise

        try:
            _write_owner(
                handle,
                {
                    "pid": os.getpid(),
                    "host": socket.gethostname(),
                    "operation": self._operation,
                    "acquired_at": time.time(),
                },
            )
        except Exception:
            try:
                handle.seek(0)
                _release_native_lock(handle)
            finally:
                handle.close()
            raise
        self._handle = handle

    def release(self) -> None:
        """Release ownership; repeated calls are harmless."""
        handle = self._handle
        if handle is None:
            return
        self._handle = None
        try:
            handle.seek(0)
            _release_native_lock(handle)
        finally:
            handle.close()

    def update_operation(self, operation: str) -> None:
        """Update diagnostic owner metadata while retaining the same OS lock."""
        handle = self._handle
        if handle is None:
            raise RuntimeError("Collection write lock is not acquired.")
        self._operation = operation
        _write_owner(
            handle,
            {
                "pid": os.getpid(),
                "host": socket.gethostname(),
                "operation": operation,
                "acquired_at": time.time(),
            },
        )

    def __enter__(self) -> CollectionWriteLock:
        self.acquire()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.release()

    def _busy_message(self, owner: dict[str, object] | None) -> str:
        detail = ""
        if owner:
            pid = owner.get("pid", "unknown")
            operation = owner.get("operation", "another write operation")
            detail = f" Process {pid} is running {operation}."
        return (
            f"Collection '{self._collection}' is busy.{detail} "
            "Wait for the active indexing or enrichment operation to finish."
        )


def _acquire_native_lock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


def _release_native_lock(handle: BinaryIO) -> None:
    if sys.platform == "win32":
        import msvcrt

        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return

    import fcntl

    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _is_lock_contention(exc: OSError) -> bool:
    return exc.errno in {
        errno.EACCES,
        errno.EAGAIN,
        errno.EBUSY,
        errno.EDEADLK,
    }


def _write_owner(handle: BinaryIO, owner: dict[str, object]) -> None:
    payload = json.dumps(owner, sort_keys=True).encode("utf-8")
    handle.seek(0)
    handle.write(payload)
    handle.truncate()
    handle.flush()
    os.fsync(handle.fileno())
    handle.seek(0)


def _read_owner(handle: BinaryIO) -> dict[str, object] | None:
    try:
        handle.seek(0)
        raw = json.loads(handle.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, dict) else None


__all__ = ["CollectionBusyError", "CollectionWriteLock"]

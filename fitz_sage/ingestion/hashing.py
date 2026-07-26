"""Canonical SHA-256 content identity for ingestion and manifests."""

from __future__ import annotations

import hashlib
from pathlib import Path


def compute_content_hash(path: str | Path) -> str:
    """Return the 64-character SHA-256 digest of a file's bytes."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if source.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(65536):
            digest.update(block)
    return digest.hexdigest()


def compute_bytes_hash(data: bytes) -> str:
    """Return the 64-character SHA-256 digest of bytes already in memory."""
    return hashlib.sha256(data).hexdigest()


__all__ = ["compute_bytes_hash", "compute_content_hash"]

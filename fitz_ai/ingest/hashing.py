# fitz_ai/ingest/hashing.py
"""
Content hashing for incremental ingestion.

Provides SHA-256 based content identity for files.
This is the single source of truth for file identity in the diff ingest system.

Design:
- Content hash is computed from raw file bytes
- Hash is used to determine if a file has changed
- Same content in different paths = same hash (content-addressable)
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union


def compute_content_hash(path: Union[str, Path]) -> str:
    """
    Compute SHA-256 hash of a file's contents.

    This is the canonical way to determine file identity for incremental ingestion.
    Two files with identical content will have identical hashes, regardless of path.

    Args:
        path: Path to the file to hash

    Returns:
        SHA-256 hash as hex string with "sha256:" prefix

    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
        IsADirectoryError: If path is a directory

    Examples:
        >>> compute_content_hash("document.txt")
        'sha256:e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855'
    """
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory: {path}")

    # Read file in chunks to handle large files efficiently
    hasher = hashlib.sha256()
    chunk_size = 65536  # 64KB chunks

    with p.open("rb") as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)

    return f"sha256:{hasher.hexdigest()}"


def compute_bytes_hash(data: bytes) -> str:
    """
    Compute SHA-256 hash of raw bytes.

    Useful for hashing content that's already in memory.

    Args:
        data: Raw bytes to hash

    Returns:
        SHA-256 hash as hex string with "sha256:" prefix
    """
    return f"sha256:{hashlib.sha256(data).hexdigest()}"


def compute_chunk_id(
    content_hash: str,
    chunk_index: int,
    parser_id: str,
    chunker_id: str,
    embedding_id: str,
) -> str:
    """
    Compute deterministic chunk ID per spec §5.2.

    The chunk ID is a SHA-256 hash of the concatenation of:
    - content_hash (file content hash)
    - chunk_index (position in file)
    - parser_id (parser version)
    - chunker_id (chunker config)
    - embedding_id (embedding config)

    This ensures:
    - Same file + same config = same chunk IDs (safe upserts)
    - Config change = new chunk IDs (triggers re-ingestion)
    - Deterministic and reproducible

    Args:
        content_hash: SHA-256 hash of file content
        chunk_index: 0-based index of chunk within file
        parser_id: Parser identifier (e.g., "md.v1")
        chunker_id: Chunker identifier (e.g., "tokens_800_120")
        embedding_id: Embedding identifier (e.g., "openai:text-embedding-3-small")

    Returns:
        SHA-256 hash as hex string (no prefix, used as vector ID)
    """
    # Build the composite key
    key = "|".join(
        [
            content_hash,
            str(chunk_index),
            parser_id,
            chunker_id,
            embedding_id,
        ]
    )

    return hashlib.sha256(key.encode("utf-8")).hexdigest()


__all__ = [
    "compute_content_hash",
    "compute_bytes_hash",
    "compute_chunk_id",
]

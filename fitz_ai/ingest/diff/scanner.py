# fitz_ai/ingest/diff/scanner.py
"""
File scanner for incremental ingestion.

Walks directories, filters by supported extensions, and computes content hashes.

This module is responsible for the "scan" phase (§7.1):
1. Walk files recursively
2. Filter by supported extensions
3. Compute content hash for each file
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Set

from fitz_ai.ingest.hashing import compute_content_hash

logger = logging.getLogger(__name__)


# Supported file extensions per spec §1
SUPPORTED_EXTENSIONS: Set[str] = {
    ".md",
    ".txt",
    ".py",
    ".pdf",
}


@dataclass(frozen=True)
class ScannedFile:
    """
    Result of scanning a single file.

    Contains all information needed for diff computation.
    """

    path: str  # Absolute path
    root: str  # Absolute path of scan root
    ext: str  # Extension (e.g., ".md")
    size_bytes: int  # File size
    mtime_epoch: float  # Modification time as Unix epoch
    content_hash: str  # SHA-256 content hash

    @property
    def relative_path(self) -> str:
        """Get path relative to root."""
        return str(Path(self.path).relative_to(self.root))


@dataclass
class ScanResult:
    """
    Result of scanning a directory.

    Contains all scanned files and any errors encountered.
    """

    root: str  # Absolute path of scan root
    files: list[ScannedFile]  # Successfully scanned files
    errors: list[tuple[str, str]]  # (path, error_message) for failed files
    skipped_extensions: dict[str, int]  # Count of skipped files by extension

    @property
    def total_scanned(self) -> int:
        """Total files successfully scanned."""
        return len(self.files)

    @property
    def total_errors(self) -> int:
        """Total files that failed to scan."""
        return len(self.errors)

    @property
    def total_skipped(self) -> int:
        """Total files skipped due to unsupported extension."""
        return sum(self.skipped_extensions.values())


class FileScanner:
    """
    Scans directories for files to ingest.

    Usage:
        scanner = FileScanner()
        result = scanner.scan("/path/to/documents")

        for file in result.files:
            print(f"{file.path}: {file.content_hash}")
    """

    def __init__(
        self,
        supported_extensions: Set[str] | None = None,
    ) -> None:
        """
        Initialize the scanner.

        Args:
            supported_extensions: Set of extensions to include. If None, uses defaults.
        """
        self._extensions = supported_extensions or SUPPORTED_EXTENSIONS

    def scan(self, root: str | Path) -> ScanResult:
        """
        Scan a directory or file.

        Args:
            root: Path to directory or single file

        Returns:
            ScanResult with all scanned files and errors
        """
        root_path = Path(root).resolve()
        root_str = str(root_path)

        files: list[ScannedFile] = []
        errors: list[tuple[str, str]] = []
        skipped: dict[str, int] = {}

        if root_path.is_file():
            # Single file
            scanned = self._scan_file(root_path, root_path.parent)
            if scanned is not None:
                files.append(scanned)
            elif root_path.suffix.lower() not in self._extensions:
                ext = root_path.suffix.lower()
                skipped[ext] = skipped.get(ext, 0) + 1
        else:
            # Directory - walk recursively
            for scanned_file, error, skip_ext in self._walk_directory(root_path):
                if scanned_file is not None:
                    files.append(scanned_file)
                elif error is not None:
                    errors.append(error)
                elif skip_ext is not None:
                    skipped[skip_ext] = skipped.get(skip_ext, 0) + 1

        logger.info(
            f"Scanned {root_str}: {len(files)} files, "
            f"{len(errors)} errors, {sum(skipped.values())} skipped"
        )

        return ScanResult(
            root=root_str,
            files=files,
            errors=errors,
            skipped_extensions=skipped,
        )

    def _walk_directory(
        self,
        root: Path,
    ) -> Iterator[tuple[ScannedFile | None, tuple[str, str] | None, str | None]]:
        """
        Walk a directory and yield scan results.

        Yields tuples of:
        - (ScannedFile, None, None) for successful scans
        - (None, (path, error), None) for errors
        - (None, None, extension) for skipped files
        """
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue

                # Skip hidden files and directories
                if any(part.startswith(".") for part in path.parts):
                    continue

                ext = path.suffix.lower()

                # Check if extension is supported
                if ext not in self._extensions:
                    yield (None, None, ext)
                    continue

                # Try to scan the file
                try:
                    scanned = self._scan_file(path, root)
                    if scanned is not None:
                        yield (scanned, None, None)
                except Exception as e:
                    yield (None, (str(path), str(e)), None)

        except PermissionError as e:
            yield (None, (str(root), f"Permission denied: {e}"), None)

    def _scan_file(self, path: Path, root: Path) -> ScannedFile | None:
        """
        Scan a single file.

        Returns None if file should be skipped.
        Raises exception on error.
        """
        ext = path.suffix.lower()

        if ext not in self._extensions:
            return None

        stat = path.stat()
        content_hash = compute_content_hash(path)

        return ScannedFile(
            path=str(path.resolve()),
            root=str(root.resolve()),
            ext=ext,
            size_bytes=stat.st_size,
            mtime_epoch=stat.st_mtime,
            content_hash=content_hash,
        )


def scan_directory(root: str | Path) -> ScanResult:
    """
    Convenience function to scan a directory.

    Args:
        root: Path to directory or single file

    Returns:
        ScanResult with all scanned files
    """
    return FileScanner().scan(root)


__all__ = [
    "SUPPORTED_EXTENSIONS",
    "ScannedFile",
    "ScanResult",
    "FileScanner",
    "scan_directory",
]

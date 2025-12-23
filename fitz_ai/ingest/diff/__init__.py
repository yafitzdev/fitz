# fitz_ai/ingest/diff/__init__.py
"""
Incremental (diff) ingestion for Fitz.

This package implements content-hash based incremental ingestion:
- Only ingest files that have changed
- Skip files that match state file (authoritative source)
- Track deletions via state file

Key components:
- Scanner: Walks directories and computes content hashes
- Differ: Computes action plan by checking state file
- Executor: Orchestrates the full pipeline

Usage:
    from fitz_ai.ingest.diff import run_diff_ingest
    from fitz_ai.ingest.state import IngestStateManager

    summary = run_diff_ingest(
        source="./documents",
        state_manager=IngestStateManager(),
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunker=chunker_plugin,
        collection="my_docs",
    )
    print(summary)  # "scanned 10, ingested 3, skipped 7, ..."
"""

from .differ import (
    Differ,
    DiffResult,
    FileCandidate,
    StateReader,
    compute_diff,
)
from .executor import (
    DiffIngestExecutor,
    Embedder,
    IngestSummary,
    Parser,
    VectorDBWriter,
    run_diff_ingest,
)
from .scanner import (
    SUPPORTED_EXTENSIONS,
    FileScanner,
    scan_directory,
    ScannedFile,
    ScanResult,
)

__all__ = [
    # Scanner
    "SUPPORTED_EXTENSIONS",
    "ScannedFile",
    "ScanResult",
    "FileScanner",
    "scan_directory",
    # Differ
    "StateReader",
    "FileCandidate",
    "DiffResult",
    "Differ",
    "compute_diff",
    # Executor
    "VectorDBWriter",
    "Embedder",
    "Parser",
    "IngestSummary",
    "DiffIngestExecutor",
    "run_diff_ingest",
]
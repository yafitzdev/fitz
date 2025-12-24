# fitz_ai/ingest/diff/__init__.py
"""
Incremental (diff) ingestion for Fitz.

This package implements content-hash AND config-aware incremental ingestion:
- Only ingest files that have changed content
- Re-ingest files when chunking/parsing/embedding config changes
- Track deletions via state file

Key components:
- Scanner: Walks directories and computes content hashes
- Differ: Computes action plan by checking state AND config
- Executor: Orchestrates the full pipeline with ChunkingRouter

Usage:
    from fitz_ai.ingest.diff import run_diff_ingest
    from fitz_ai.ingest.state import IngestStateManager
    from fitz_ai.ingest.chunking import ChunkingRouter

    router = ChunkingRouter.from_config(config)

    summary = run_diff_ingest(
        source="./documents",
        state_manager=IngestStateManager(),
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunking_router=router,
        collection="my_docs",
        embedding_id="cohere:embed-english-v3.0",
    )
    print(summary)
"""

from fitz_ai.ingest.diff.differ import (
    ConfigProvider,
    Differ,
    DiffResult,
    FileCandidate,
    ReingestReason,
    StateReader,
    compute_diff,
)
from fitz_ai.ingest.diff.executor import (
    DiffIngestExecutor,
    Embedder,
    IngestSummary,
    Parser,
    VectorDBWriter,
    run_diff_ingest,
)
from fitz_ai.ingest.diff.scanner import (
    SUPPORTED_EXTENSIONS,
    FileScanner,
    ScannedFile,
    ScanResult,
)

__all__ = [
    # Scanner
    "SUPPORTED_EXTENSIONS",
    "ScannedFile",
    "ScanResult",
    "FileScanner",
    # Differ
    "StateReader",
    "ConfigProvider",
    "FileCandidate",
    "DiffResult",
    "ReingestReason",
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

# tests/unit/test_ingest_real_embedder.py
"""
Test ingestion with the configured embedder to diagnose slowness.

Uses local Ollama embedder from tests/test_config.yaml for fast execution.

Run with: pytest tests/unit/test_ingest_real_embedder.py -v -s --log-cli-level=INFO

NOTE: These tests require a running Ollama instance and are skipped if unavailable.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

# Mark as tier3 (requires real embeddings service) and integration
pytestmark = [pytest.mark.tier3, pytest.mark.integration, pytest.mark.embeddings]

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument
from fitz_sage.ingestion.source.base import SourceFile


def _ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        import httpx

        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False


# Skip all tests in this module if Ollama is not available
pytestmark = pytest.mark.skipif(
    not _ollama_available(), reason="Ollama not available (requires local Ollama instance)"
)

# Enable all logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class TracingVectorDBWriter:
    """Vector DB writer that traces all calls."""

    def __init__(self):
        self.upsert_calls = 0
        self.total_points = 0

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        self.upsert_calls += 1
        self.total_points += len(points)
        logger.info(f"[VDB] upsert #{self.upsert_calls}: {len(points)} points")

    def mark_deleted(self, collection: str, source_path: str) -> int:
        return 1


class TracingParserRouter:
    """Parser router that traces calls and returns real-ish content."""

    def __init__(self):
        self.parse_calls = 0

    def parse(self, source_file: SourceFile) -> ParsedDocument:
        self.parse_calls += 1
        path = str(source_file.local_path)
        # Return content that creates multiple chunks
        content = f"# {Path(path).name}\n\n" + ("This is test content for embedding. " * 100)
        return ParsedDocument(
            source=source_file.uri,
            elements=[DocumentElement(type=ElementType.TEXT, content=content)],
            metadata={"source_file": path},
        )

    def get_parser_id(self, ext: str) -> str:
        return f"tracing:{ext[1:] if ext.startswith('.') else ext}:v1"


@pytest.fixture
def test_files(tmp_path: Path):
    """Create test files."""
    files = []
    for i in range(10):
        f = tmp_path / f"file_{i}.py"
        f.write_text(f"# file {i}\n" + "x = 1\n" * 50)
        files.append(f)
    return tmp_path, files


def test_with_real_embedder(tmp_path: Path, test_files):
    """
    Test with the configured embedder from test_config.yaml.

    Run: pytest tests/unit/test_ingest_real_embedder.py::test_with_real_embedder -v -s --log-cli-level=INFO
    """
    from fitz_sage.ingestion.chunking.plugins.default.recursive import RecursiveChunker
    from fitz_sage.ingestion.chunking.router import ChunkingRouter
    from fitz_sage.ingestion.diff.executor import DiffIngestExecutor
    from fitz_sage.ingestion.state import IngestStateManager

    source_path, files = test_files

    print(f"\n{'=' * 70}")
    print(f"TEST: Real embedder with {len(files)} files")
    print(f"{'=' * 70}\n")

    # Setup
    state_path = tmp_path / "state" / "ingest.json"
    state_manager = IngestStateManager(state_path)
    state_manager.load()

    router = ChunkingRouter(
        chunker_map={},
        default_chunker=RecursiveChunker(chunk_size=500, chunk_overlap=50),
        warn_on_fallback=False,
    )

    # Get embedder from test config (local Ollama by default)
    from tests.conftest import get_test_embedder

    print("[TEST] Creating embedder from test config...")
    t0 = time.perf_counter()
    embedder = get_test_embedder()
    print(f"[TEST] Embedder created in {time.perf_counter() - t0:.2f}s")

    writer = TracingVectorDBWriter()
    parser_router = TracingParserRouter()

    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        parser_router=parser_router,
        chunking_router=router,
        collection="test_collection",
        embedding_id="local_ollama:nomic-embed-text",
    )

    # Run ingestion
    print("\n[TEST] Starting ingestion...")
    t_start = time.perf_counter()
    summary = executor.run(source_path)
    t_total = time.perf_counter() - t_start

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Files:          {len(files)}")
    print(f"  Chunks:         {writer.total_points}")
    print(f"  Total time:     {t_total:.2f}s")
    print(f"  Time per chunk: {t_total / max(1, writer.total_points) * 1000:.0f}ms")
    print(f"  Summary:        {summary}")
    print(f"{'=' * 70}")


def test_embedder_batch_directly(tmp_path: Path):
    """
    Test the embedder's embed_batch directly to isolate the issue.

    Run: pytest tests/unit/test_ingest_real_embedder.py::test_embedder_batch_directly -v -s --log-cli-level=INFO
    """
    from tests.conftest import get_test_embedder

    print(f"\n{'=' * 70}")
    print("TEST: Direct embed_batch() call")
    print(f"{'=' * 70}\n")

    embedder = get_test_embedder()

    # Create test texts
    texts = [f"This is test document number {i}. " * 20 for i in range(50)]
    print(
        f"[TEST] Created {len(texts)} texts, avg length: {sum(len(t) for t in texts) // len(texts)} chars"
    )

    # Call embed_batch directly
    print("\n[TEST] Calling embed_batch()...")
    t0 = time.perf_counter()
    vectors = embedder.embed_batch(texts)
    t1 = time.perf_counter()

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  Texts:          {len(texts)}")
    print(f"  Vectors:        {len(vectors)}")
    print(f"  Vector dim:     {len(vectors[0]) if vectors else 'N/A'}")
    print(f"  Total time:     {t1 - t0:.2f}s")
    print(f"  Time per text:  {(t1 - t0) / len(texts) * 1000:.0f}ms")
    print(f"{'=' * 70}")


def test_embedder_single_vs_batch(tmp_path: Path):
    """
    Compare single embed() vs embed_batch() to verify batching helps.

    Run: pytest tests/unit/test_ingest_real_embedder.py::test_embedder_single_vs_batch -v -s --log-cli-level=INFO
    """
    from tests.conftest import get_test_embedder

    print(f"\n{'=' * 70}")
    print("TEST: Single embed() vs embed_batch()")
    print(f"{'=' * 70}\n")

    embedder = get_test_embedder()

    texts = [f"Test document {i}. " * 10 for i in range(10)]

    # Test single embed calls
    print("[TEST] Testing 10 individual embed() calls...")
    t0 = time.perf_counter()
    for text in texts:
        embedder.embed(text)
    t_single = time.perf_counter() - t0
    print(f"[TEST] Single calls: {t_single:.2f}s ({t_single / len(texts) * 1000:.0f}ms each)")

    # Test batch embed
    print("\n[TEST] Testing 1 embed_batch() call with 10 texts...")
    t0 = time.perf_counter()
    embedder.embed_batch(texts)
    t_batch = time.perf_counter() - t0
    print(f"[TEST] Batch call: {t_batch:.2f}s ({t_batch / len(texts) * 1000:.0f}ms per text)")

    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")
    print(f"  10 single embed() calls: {t_single:.2f}s")
    print(f"  1 embed_batch() call:    {t_batch:.2f}s")
    print(f"  Speedup:                 {t_single / t_batch:.1f}x")
    print(f"{'=' * 70}")

    assert t_batch < t_single, "Batch should be faster than individual calls!"

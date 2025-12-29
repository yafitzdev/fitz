# tests/test_ingest_real_embedder.py
"""
Test ingestion with the REAL Cohere embedder to diagnose slowness.

Run with: pytest tests/test_ingest_real_embedder.py -v -s --log-cli-level=INFO
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import pytest

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


class TracingParser:
    """Parser that traces calls and returns real-ish content."""

    def __init__(self):
        self.parse_calls = 0

    def parse(self, path: str) -> str:
        self.parse_calls += 1
        # Return content that creates multiple chunks
        return f"# {Path(path).name}\n\n" + ("This is test content for embedding. " * 100)


@pytest.fixture
def test_files(tmp_path: Path):
    """Create test files."""
    files = []
    for i in range(10):
        f = tmp_path / f"file_{i}.py"
        f.write_text(f"# file {i}\n" + "x = 1\n" * 50)
        files.append(f)
    return tmp_path, files


def test_with_real_cohere_embedder(tmp_path: Path, test_files):
    """
    Test with the REAL Cohere embedder.

    Run: pytest tests/test_ingest_real_embedder.py::test_with_real_cohere_embedder -v -s --log-cli-level=INFO
    """
    import os

    from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker
    from fitz_ai.ingestion.chunking.router import ChunkingRouter
    from fitz_ai.ingestion.diff.executor import DiffIngestExecutor
    from fitz_ai.ingestion.state import IngestStateManager

    # Check for API key
    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set - skipping real API test")

    source_path, files = test_files

    print(f"\n{'=' * 70}")
    print(f"TEST: Real Cohere embedder with {len(files)} files")
    print(f"{'=' * 70}\n")

    # Setup
    state_path = tmp_path / "state" / "ingest.json"
    state_manager = IngestStateManager(state_path)
    state_manager.load()

    router = ChunkingRouter(
        chunker_map={},
        default_chunker=SimpleChunker(chunk_size=500, chunk_overlap=50),
        warn_on_fallback=False,
    )

    # Get REAL Cohere embedder
    from fitz_ai.llm.registry import get_llm_plugin

    print("[TEST] Creating real Cohere embedder...")
    t0 = time.perf_counter()
    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name="cohere",
        model="embed-english-v3.0",
    )
    print(f"[TEST] Embedder created in {time.perf_counter() - t0:.2f}s")

    writer = TracingVectorDBWriter()
    parser = TracingParser()

    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunking_router=router,
        collection="test_collection",
        embedding_id="cohere:embed-english-v3.0",
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

    Run: pytest tests/test_ingest_real_embedder.py::test_embedder_batch_directly -v -s --log-cli-level=INFO
    """
    import os

    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set")

    from fitz_ai.llm.registry import get_llm_plugin

    print(f"\n{'=' * 70}")
    print("TEST: Direct embed_batch() call")
    print(f"{'=' * 70}\n")

    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name="cohere",
        model="embed-english-v3.0",
    )

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

    Run: pytest tests/test_ingest_real_embedder.py::test_embedder_single_vs_batch -v -s --log-cli-level=INFO
    """
    import os

    if not os.environ.get("COHERE_API_KEY"):
        pytest.skip("COHERE_API_KEY not set")

    from fitz_ai.llm.registry import get_llm_plugin

    print(f"\n{'=' * 70}")
    print("TEST: Single embed() vs embed_batch()")
    print(f"{'=' * 70}\n")

    embedder = get_llm_plugin(
        plugin_type="embedding",
        plugin_name="cohere",
        model="embed-english-v3.0",
    )

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

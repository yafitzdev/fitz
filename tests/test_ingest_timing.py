# tests/test_ingest_timing.py
"""
Diagnostic test to pinpoint ingestion slowness.

Run with: pytest tests/test_ingest_timing.py -v -s
"""

import time
from pathlib import Path
from typing import Any, Dict, List

import pytest


class TimingTracker:
    """Track timing of operations."""

    def __init__(self):
        self.events: List[tuple] = []
        self.start_time = time.perf_counter()

    def log(self, event: str, details: str = ""):
        elapsed = time.perf_counter() - self.start_time
        self.events.append((elapsed, event, details))
        print(f"[{elapsed:8.3f}s] {event}: {details}")

    def summary(self):
        print("\n" + "=" * 60)
        print("TIMING SUMMARY")
        print("=" * 60)
        prev_time = 0
        for elapsed, event, details in self.events:
            delta = elapsed - prev_time
            print(f"  +{delta:6.3f}s | {event}: {details}")
            prev_time = elapsed
        print("=" * 60)
        print(f"  TOTAL: {self.events[-1][0]:.3f}s")
        print("=" * 60)


class TracingEmbedder:
    """Embedder that traces all calls."""

    def __init__(self, tracker: TimingTracker, dim: int = 4, delay: float = 0.0):
        self._dim = dim
        self._delay = delay
        self._tracker = tracker
        self.embed_calls = 0
        self.embed_batch_calls = 0
        self.total_texts_embedded = 0

    def embed(self, text: str) -> List[float]:
        self.embed_calls += 1
        self.total_texts_embedded += 1
        self._tracker.log("embed()", f"call #{self.embed_calls}, text_len={len(text)}")
        if self._delay:
            time.sleep(self._delay)
        return [0.1] * self._dim

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        self.embed_batch_calls += 1
        self.total_texts_embedded += len(texts)
        self._tracker.log(
            "embed_batch()",
            f"call #{self.embed_batch_calls}, num_texts={len(texts)}, "
            f"total_chars={sum(len(t) for t in texts)}",
        )
        if self._delay:
            time.sleep(self._delay * len(texts) * 0.1)  # Simulate batch being faster
        return [[0.1] * self._dim for _ in texts]


class TracingVectorDBWriter:
    """Vector DB writer that traces all calls."""

    def __init__(self, tracker: TimingTracker):
        self._tracker = tracker
        self.upsert_calls = 0
        self.total_points = 0

    def upsert(
        self, collection: str, points: List[Dict[str, Any]], defer_persist: bool = False
    ) -> None:
        self.upsert_calls += 1
        self.total_points += len(points)
        self._tracker.log(
            "upsert()",
            f"call #{self.upsert_calls}, collection={collection}, points={len(points)}",
        )

    def mark_deleted(self, collection: str, source_path: str) -> int:
        self._tracker.log("mark_deleted()", f"collection={collection}, path={source_path}")
        return 1


class TracingParser:
    """Parser that traces all calls."""

    def __init__(self, tracker: TimingTracker, content_size: int = 500):
        self._tracker = tracker
        self._content_size = content_size
        self.parse_calls = 0

    def parse(self, path: str) -> str:
        self.parse_calls += 1
        self._tracker.log("parse()", f"call #{self.parse_calls}, path={Path(path).name}")
        # Return content that will create multiple chunks
        return f"# File: {path}\n\n" + ("Lorem ipsum dolor sit amet. " * 50) * (
            self._content_size // 100
        )


@pytest.fixture
def timing_tracker():
    return TimingTracker()


@pytest.fixture
def test_files(tmp_path: Path):
    """Create test files of varying sizes."""
    files = []

    # Create 10 small files
    for i in range(10):
        f = tmp_path / f"small_{i}.py"
        f.write_text(f"# small file {i}\n" + "x = 1\n" * 50)
        files.append(f)

    # Create 5 medium files
    for i in range(5):
        f = tmp_path / f"medium_{i}.py"
        f.write_text(f"# medium file {i}\n" + "def func():\n    pass\n" * 200)
        files.append(f)

    # Create 2 large files
    for i in range(2):
        f = tmp_path / f"large_{i}.py"
        f.write_text(f"# large file {i}\n" + "class Foo:\n    def bar(self):\n        pass\n" * 500)
        files.append(f)

    return tmp_path, files


def test_ingest_timing_detailed(tmp_path: Path, timing_tracker: TimingTracker, test_files):
    """
    Detailed timing test for ingestion pipeline.

    Run with: pytest tests/test_ingest_timing.py::test_ingest_timing_detailed -v -s
    """
    from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker
    from fitz_ai.ingestion.chunking.router import ChunkingRouter
    from fitz_ai.ingestion.diff.executor import DiffIngestExecutor
    from fitz_ai.ingestion.state import IngestStateManager

    source_path, files = test_files

    print(f"\n{'=' * 60}")
    print(f"TEST: Ingesting {len(files)} files from {source_path}")
    print(f"{'=' * 60}\n")

    # Setup
    timing_tracker.log("SETUP", "Creating components")

    state_path = tmp_path / "state" / "ingest.json"
    state_manager = IngestStateManager(state_path)
    state_manager.load()

    router = ChunkingRouter(
        chunker_map={},
        default_chunker=SimpleChunker(chunk_size=500, chunk_overlap=50),
        warn_on_fallback=False,
    )

    embedder = TracingEmbedder(timing_tracker, delay=0.01)  # 10ms simulated API latency
    writer = TracingVectorDBWriter(timing_tracker)
    parser = TracingParser(timing_tracker)

    timing_tracker.log("SETUP", "Components created")

    # Create executor
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunking_router=router,
        collection="test_collection",
        embedding_id="test:embedding",
    )

    timing_tracker.log("EXECUTOR", "Created")

    # Run ingestion
    timing_tracker.log("INGEST", "Starting run()")
    summary = executor.run(source_path)
    timing_tracker.log("INGEST", f"Completed: {summary}")

    # Print summary
    timing_tracker.summary()

    print(f"\n{'=' * 60}")
    print("CALL STATISTICS")
    print(f"{'=' * 60}")
    print(f"  Parser.parse() calls:      {parser.parse_calls}")
    print(f"  Embedder.embed() calls:    {embedder.embed_calls}")
    print(f"  Embedder.embed_batch() calls: {embedder.embed_batch_calls}")
    print(f"  Total texts embedded:      {embedder.total_texts_embedded}")
    print(f"  VectorDB.upsert() calls:   {writer.upsert_calls}")
    print(f"  Total points upserted:     {writer.total_points}")
    print(f"{'=' * 60}")

    # Assertions to verify batching is working
    assert (
        embedder.embed_calls == 0
    ), f"embed() should not be called, but was called {embedder.embed_calls} times"
    assert embedder.embed_batch_calls >= 1, "embed_batch() should be called at least once"

    # If we have 17 files and batching works, we should have far fewer embed_batch calls than files
    if len(files) > 5:
        assert embedder.embed_batch_calls < len(files), (
            f"Batching not working: {embedder.embed_batch_calls} embed_batch calls for {len(files)} files. "
            f"Should be fewer calls if cross-file batching is working."
        )

    print("\n[PASS] Batching is working correctly!")
    print(f"  - {embedder.embed_batch_calls} embed_batch calls for {len(files)} files")
    print(f"  - {embedder.total_texts_embedded} total chunks embedded")


def test_ingest_with_real_embedder_mock(tmp_path: Path, timing_tracker: TimingTracker, test_files):
    """
    Test with a mock that simulates real API latency.

    Run with: pytest tests/test_ingest_timing.py::test_ingest_with_real_embedder_mock -v -s
    """
    from fitz_ai.ingestion.chunking.plugins.default.simple import SimpleChunker
    from fitz_ai.ingestion.chunking.router import ChunkingRouter
    from fitz_ai.ingestion.diff.executor import DiffIngestExecutor
    from fitz_ai.ingestion.state import IngestStateManager

    source_path, files = test_files

    print(f"\n{'=' * 60}")
    print("TEST: Simulating real API latency")
    print(f"{'=' * 60}\n")

    state_path = tmp_path / "state" / "ingest.json"
    state_manager = IngestStateManager(state_path)
    state_manager.load()

    router = ChunkingRouter(
        chunker_map={},
        default_chunker=SimpleChunker(chunk_size=500, chunk_overlap=50),
        warn_on_fallback=False,
    )

    # Simulate realistic API latency
    # Real Cohere embed API: ~100-300ms per batch
    embedder = TracingEmbedder(timing_tracker, delay=0.1)  # 100ms per single embed
    writer = TracingVectorDBWriter(timing_tracker)
    parser = TracingParser(timing_tracker)

    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunking_router=router,
        collection="test_collection",
        embedding_id="test:embedding",
    )

    timing_tracker.log("INGEST", "Starting")
    summary = executor.run(source_path)
    timing_tracker.log("INGEST", f"Done: {summary}")

    timing_tracker.summary()

    print(f"\n{'=' * 60}")
    print("LATENCY ANALYSIS")
    print(f"{'=' * 60}")

    # Calculate what time WOULD have been with per-file batching
    per_file_time = len(files) * 0.1  # 100ms per file

    print(f"  Files processed:           {len(files)}")
    print(f"  embed_batch() calls:       {embedder.embed_batch_calls}")
    print(f"  Chunks embedded:           {embedder.total_texts_embedded}")
    print(f"  Estimated per-file time:   {per_file_time:.2f}s (if 1 API call per file)")
    print(f"  Actual batch calls time:   ~{embedder.embed_batch_calls * 0.01:.2f}s")
    print(f"{'=' * 60}")

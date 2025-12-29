# tests/test_ingest_cli_mirror.py
"""
Exact mirror of the CLI ingest process with detailed logging.

Run with: pytest tests/test_ingest_cli_mirror.py -v -s --log-cli-level=DEBUG

NOTE: This is an integration test that requires a real config file.
It is skipped in CI when no config is present.
"""

import logging
import time
from pathlib import Path

import pytest

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


def log_timing(phase: str, start: float, details: str = ""):
    elapsed = time.perf_counter() - start
    print(f"[TIMING] {phase}: {elapsed:.3f}s {details}")
    return time.perf_counter()


@pytest.mark.integration
def test_ingest_cli_mirror():
    """
    Mirror the exact CLI ingest process.

    Run: pytest tests/test_ingest_cli_mirror.py::test_ingest_cli_mirror -v -s --log-cli-level=DEBUG
    """
    from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
    from fitz_ai.core.paths import FitzPaths

    # Skip if no config file exists (e.g., in CI)
    try:
        config_path = FitzPaths.config()
        if not config_path.exists():
            pytest.skip("No config file found - run 'fitz init' first")
    except Exception:
        pytest.skip("No config file found - run 'fitz init' first")

    T0 = time.perf_counter()
    t = T0

    print("\n" + "=" * 80)
    print("CLI INGEST MIRROR TEST - DETAILED TIMING")
    print("=" * 80 + "\n")

    # =========================================================================
    # PHASE 1: Load config (mirrors _load_config)
    # =========================================================================
    print("[PHASE 1] Loading config...")
    t = time.perf_counter()

    try:
        config = load_config_dict(FitzPaths.config())
    except ConfigNotFoundError:
        pytest.skip("Config file not found - run 'fitz init' first")
    t = log_timing("Config loaded", t, f"keys: {list(config.keys())}")

    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    embedding_kwargs = config.get("embedding", {}).get("kwargs", {})
    embedding_model = embedding_kwargs.get("model", "embed-english-v3.0")
    embedding_id = f"{embedding_plugin}:{embedding_model}"

    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    vector_db_kwargs = config.get("vector_db", {}).get("kwargs", {})

    print(f"  Embedding: {embedding_id}")
    print(f"  Vector DB: {vector_db_plugin}")
    print(f"  Embedding kwargs: {embedding_kwargs}")
    print(f"  Vector DB kwargs: {vector_db_kwargs}")

    # =========================================================================
    # PHASE 2: Create components (mirrors CLI initialization)
    # =========================================================================
    print("\n[PHASE 2] Creating components...")

    # State manager
    t = time.perf_counter()
    from fitz_ai.ingestion.state import IngestStateManager

    state_manager = IngestStateManager()
    state_manager.load()
    t = log_timing("State manager", t)

    # Parser
    t = time.perf_counter()
    from fitz_ai.ingestion.reader.registry import get_ingest_plugin

    IngestPluginCls = get_ingest_plugin("local")
    ingest_plugin = IngestPluginCls()
    t = log_timing("Parser plugin", t)

    # Wrap parser like CLI does
    class ParserAdapter:
        def __init__(self, plugin):
            self._plugin = plugin
            self.parse_count = 0
            self.parse_times = []

        def parse(self, path: str) -> str:
            t0 = time.perf_counter()
            self.parse_count += 1
            docs = list(self._plugin.ingest(path, kwargs={}))
            elapsed = time.perf_counter() - t0
            self.parse_times.append(elapsed)
            if self.parse_count <= 5 or self.parse_count % 10 == 0:
                print(f"    [PARSE] #{self.parse_count} {Path(path).name}: {elapsed:.3f}s")
            return docs[0].content if docs else ""

    parser = ParserAdapter(ingest_plugin)

    # Chunking router
    t = time.perf_counter()
    from fitz_ai.engines.classic_rag.config import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )
    from fitz_ai.ingestion.chunking.router import ChunkingRouter

    chunking = config.get("chunking", {})
    default_cfg = chunking.get("default", {})
    default = ExtensionChunkerConfig(
        plugin_name=default_cfg.get("plugin_name", "simple"),
        kwargs=default_cfg.get("kwargs", {"chunk_size": 1000, "chunk_overlap": 0}),
    )
    by_extension = {}
    for ext, ext_cfg in chunking.get("by_extension", {}).items():
        by_extension[ext] = ExtensionChunkerConfig(
            plugin_name=ext_cfg.get("plugin_name", "simple"),
            kwargs=ext_cfg.get("kwargs", {}),
        )
    router_config = ChunkingRouterConfig(
        default=default,
        by_extension=by_extension,
        warn_on_fallback=chunking.get("warn_on_fallback", True),
    )
    chunking_router = ChunkingRouter.from_config(router_config)
    t = log_timing("Chunking router", t, f"extensions: {list(by_extension.keys())}")

    # Embedder
    t = time.perf_counter()
    from fitz_ai.llm.registry import get_llm_plugin

    embedder = get_llm_plugin(
        plugin_type="embedding", plugin_name=embedding_plugin, **embedding_kwargs
    )
    t = log_timing("Embedder", t, f"{embedding_plugin}")

    # Vector DB
    t = time.perf_counter()
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    vector_client = get_vector_db_plugin(vector_db_plugin, **vector_db_kwargs)
    t = log_timing("Vector DB", t, f"{vector_db_plugin}")

    # Wrap vector DB like CLI does
    class VectorDBWriterAdapter:
        def __init__(self, client):
            self._client = client
            self.upsert_count = 0
            self.upsert_times = []
            self.total_points = 0

        def upsert(self, collection: str, points, defer_persist: bool = False) -> None:
            t0 = time.perf_counter()
            self.upsert_count += 1
            self.total_points += len(points)
            try:
                self._client.upsert(collection, points, defer_persist=defer_persist)
            except TypeError:
                self._client.upsert(collection, points)
            elapsed = time.perf_counter() - t0
            self.upsert_times.append(elapsed)
            if self.upsert_count <= 5 or self.upsert_count % 10 == 0:
                print(
                    f"    [UPSERT] #{self.upsert_count} {len(points)} points: {elapsed:.3f}s (defer={defer_persist})"
                )

        def flush(self) -> None:
            t0 = time.perf_counter()
            if hasattr(self._client, "flush"):
                self._client.flush()
            print(f"    [FLUSH] Persisted to disk: {time.perf_counter() - t0:.3f}s")

    writer = VectorDBWriterAdapter(vector_client)

    # =========================================================================
    # PHASE 3: Create test files
    # =========================================================================
    print("\n[PHASE 3] Creating test files...")
    t = time.perf_counter()

    import tempfile

    test_dir = Path(tempfile.mkdtemp(prefix="fitz_test_"))

    # Create files that mirror a real codebase
    for i in range(20):
        (test_dir / f"module_{i}.py").write_text(
            f'# module_{i}.py\n"""Module {i} docstring."""\n\n'
            + "import os\nimport sys\n\n"
            + f"class Class{i}:\n    '''Class docstring.'''\n    def method(self):\n        pass\n\n"
            * 5
            + f"def function_{i}():\n    '''Function docstring.'''\n    return {i}\n\n" * 10
        )

    for i in range(5):
        (test_dir / f"readme_{i}.md").write_text(
            f"# README {i}\n\n"
            + "## Overview\n\nThis is a test markdown file.\n\n" * 20
            + "## Details\n\n"
            + "Lorem ipsum dolor sit amet. " * 100
        )

    t = log_timing("Test files created", t, f"dir: {test_dir}, 25 files")

    # =========================================================================
    # PHASE 4: Create executor and run ingestion
    # =========================================================================
    print("\n[PHASE 4] Running ingestion...")

    from fitz_ai.ingestion.diff.executor import DiffIngestExecutor

    t = time.perf_counter()
    executor = DiffIngestExecutor(
        state_manager=state_manager,
        vector_db_writer=writer,
        embedder=embedder,
        parser=parser,
        chunking_router=chunking_router,
        collection="test_mirror",
        embedding_id=embedding_id,
        enrichment_pipeline=None,  # No enrichment
    )
    t = log_timing("Executor created", t)

    # Run ingestion with progress callback
    t = time.perf_counter()
    t_ingest_start = t

    def on_progress(current, total, path):
        if current == 1 or current == total or current % 5 == 0:
            print(
                f"    [PROGRESS] {current}/{total} - {Path(path).name if path != 'Done' else 'Done'}"
            )

    summary = executor.run(test_dir, force=True, on_progress=on_progress)

    t_ingest_end = time.perf_counter()
    t = log_timing("Ingestion complete", t_ingest_start)

    # =========================================================================
    # PHASE 5: Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    total_time = time.perf_counter() - T0

    print("\n[SUMMARY]")
    print(f"  Files scanned:    {summary.scanned}")
    print(f"  Files ingested:   {summary.ingested}")
    print(f"  Files skipped:    {summary.skipped}")
    print(f"  Errors:           {summary.errors}")
    print(f"  Total chunks:     {writer.total_points}")

    print("\n[TIMING BREAKDOWN]")
    print(f"  Total time:       {total_time:.2f}s")
    print(f"  Ingestion time:   {t_ingest_end - t_ingest_start:.2f}s")

    if parser.parse_times:
        print("\n[PARSING]")
        print(f"  Parse calls:      {parser.parse_count}")
        print(f"  Total parse time: {sum(parser.parse_times):.2f}s")
        print(
            f"  Avg per file:     {sum(parser.parse_times) / len(parser.parse_times) * 1000:.0f}ms"
        )

    if writer.upsert_times:
        print("\n[VECTOR DB]")
        print(f"  Upsert calls:     {writer.upsert_count}")
        print(f"  Total upsert time:{sum(writer.upsert_times):.2f}s")
        print(
            f"  Avg per upsert:   {sum(writer.upsert_times) / len(writer.upsert_times) * 1000:.0f}ms"
        )
        print(f"  Total points:     {writer.total_points}")

    print("\n[EMBEDDING]")
    print("  (See [EMBED] logs above for batch details)")

    if summary.errors > 0:
        print("\n[ERRORS]")
        for err in summary.error_details[:10]:
            print(f"  - {err}")

    print("\n" + "=" * 80)

    # Cleanup
    import shutil

    shutil.rmtree(test_dir, ignore_errors=True)


if __name__ == "__main__":
    test_ingest_cli_mirror()

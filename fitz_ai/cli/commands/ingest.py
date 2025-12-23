# fitz_ai/cli/commands/ingest.py
"""
Document ingestion with incremental (diff) support.

Usage:
    fitz ingest              # Interactive mode - prompts for source/collection
    fitz ingest ./docs       # Ingest specific directory
    fitz ingest ./docs -y    # Non-interactive with defaults

Chunking configuration is loaded from fitz.yaml (set via fitz init).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from fitz_ai.cli.ui import RICH, console, ui
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config() -> dict:
    """Load config or exit with helpful message."""
    try:
        return load_config_dict(FitzPaths.config())
    except ConfigNotFoundError:
        ui.error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)


def _build_chunking_router_config(config: dict):
    """
    Build ChunkingRouterConfig from fitz.yaml config.

    Expected config structure:
        chunking:
          default:
            plugin_name: simple
            kwargs:
              chunk_size: 1000
              chunk_overlap: 0
          by_extension:
            .md:
              plugin_name: markdown
              kwargs: {...}
          warn_on_fallback: true
    """
    from fitz_ai.ingest.config.schema import (
        ChunkingRouterConfig,
        ExtensionChunkerConfig,
    )

    chunking = config.get("chunking", {})

    # Build default config
    default_cfg = chunking.get("default", {})
    default = ExtensionChunkerConfig(
        plugin_name=default_cfg.get("plugin_name", "simple"),
        kwargs=default_cfg.get("kwargs", {"chunk_size": 1000, "chunk_overlap": 0}),
    )

    # Build per-extension configs
    by_extension = {}
    for ext, ext_cfg in chunking.get("by_extension", {}).items():
        by_extension[ext] = ExtensionChunkerConfig(
            plugin_name=ext_cfg.get("plugin_name", "simple"),
            kwargs=ext_cfg.get("kwargs", {}),
        )

    return ChunkingRouterConfig(
        default=default,
        by_extension=by_extension,
        warn_on_fallback=chunking.get("warn_on_fallback", True),
    )


# =============================================================================
# Adapter Classes
# =============================================================================


class ParserAdapter:
    """Adapts ingestion plugin to Parser protocol."""

    def __init__(self, plugin):
        self._plugin = plugin

    def parse(self, path: str) -> str:
        """Parse a file and return its text content."""
        docs = list(self._plugin.ingest(path, kwargs={}))
        if not docs:
            return ""
        return docs[0].content


class VectorDBWriterAdapter:
    """Adapts vector DB client to VectorDBWriter protocol."""

    def __init__(self, client):
        self._client = client

    def upsert(self, collection: str, points: List[Dict[str, Any]]) -> None:
        """Upsert points into collection."""
        self._client.upsert(collection, points)


# =============================================================================
# Main Command
# =============================================================================


def command(
    source: Optional[str] = typer.Argument(
        None,
        help="Source path (file or directory). Prompts if not provided.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-c",
        help="Collection name. Uses config default if not provided.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Non-interactive mode, use defaults.",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Force re-ingest all files, ignoring state.",
    ),
) -> None:
    """
    Ingest documents into the vector database.

    Chunking configuration is loaded from fitz.yaml (set via fitz init).

    Default behavior (incremental):
    - Skip unchanged files (same content AND same config)
    - Re-ingest files when chunking config changes
    - Only ingest new or modified files
    - Mark deleted files in vector DB

    Use --force to re-ingest everything regardless of state.

    Examples:
        fitz ingest              # Interactive mode
        fitz ingest ./docs       # Ingest specific directory
        fitz ingest ./docs -y    # Non-interactive with defaults
        fitz ingest --force      # Force re-ingest everything
    """
    from fitz_ai.ingest.chunking.router import ChunkingRouter
    from fitz_ai.ingest.diff import run_diff_ingest
    from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
    from fitz_ai.ingest.state import IngestStateManager
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # =========================================================================
    # Load config
    # =========================================================================

    config = _load_config()

    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    embedding_model = config.get("embedding", {}).get("kwargs", {}).get(
        "model", "embed-english-v3.0"
    )
    embedding_id = f"{embedding_plugin}:{embedding_model}"

    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    default_collection = config.get("retrieval", {}).get("collection", "default")

    # Get chunking config from fitz.yaml
    chunking_config = config.get("chunking", {})
    default_chunker = chunking_config.get("default", {}).get("plugin_name", "simple")
    chunk_size = chunking_config.get("default", {}).get("kwargs", {}).get("chunk_size", 1000)
    chunk_overlap = chunking_config.get("default", {}).get("kwargs", {}).get("chunk_overlap", 0)

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Ingest", "Feed your documents into the vector database")
    if force:
        ui.warning("Force mode: will re-ingest all files")
    else:
        ui.info("Incremental mode: skipping unchanged files")
    ui.info(f"Embedding: {embedding_id}")
    ui.info(f"Vector DB: {vector_db_plugin}")
    ui.info(f"Chunking: {default_chunker} (size={chunk_size}, overlap={chunk_overlap})")
    print()

    # =========================================================================
    # Interactive Prompts (only source and collection)
    # =========================================================================

    if non_interactive:
        if source is None:
            ui.error("Source path required in non-interactive mode.")
            ui.info("Usage: fitz ingest ./docs -y")
            raise typer.Exit(1)

        if collection is None:
            collection = default_collection

        ui.info(f"Source: {source}")
        ui.info(f"Collection: {collection}")

    else:
        if source is None:
            source = ui.prompt_path("Source path (file or directory)", ".")

        if collection is None:
            collection = ui.prompt_text("Collection name", default_collection)

    print()

    # =========================================================================
    # Initialize components
    # =========================================================================

    ui.step(1, 3, "Initializing...")

    try:
        # State manager
        state_manager = IngestStateManager()
        state_manager.load()

        # Ingest plugin (for parsing)
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        parser = ParserAdapter(ingest_plugin)

        # Build chunking router from config
        router_config = _build_chunking_router_config(config)
        chunking_router = ChunkingRouter.from_config(router_config)

        ui.info(f"Router: {chunking_router}")

        # Embedder
        embedder = get_llm_plugin(plugin_type="embedding", plugin_name=embedding_plugin)

        # Vector DB writer
        vector_client = get_vector_db_plugin(vector_db_plugin)
        writer = VectorDBWriterAdapter(vector_client)

    except Exception as e:
        ui.error(f"Failed to initialize: {e}")
        raise typer.Exit(1)

    ui.success("Initialized")

    # =========================================================================
    # Run diff ingestion
    # =========================================================================

    ui.step(2, 3, "Scanning files...")

    try:
        summary = run_diff_ingest(
            source=source,
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser=parser,
            chunking_router=chunking_router,
            collection=collection,
            embedding_id=embedding_id,
            force=force,
        )
    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        logger.exception("Ingestion error")
        raise typer.Exit(1)

    # =========================================================================
    # Summary
    # =========================================================================

    ui.step(3, 3, "Complete!")
    print()

    if RICH:
        from rich.table import Table

        table = Table(show_header=False, box=None, padding=(0, 2))
        table.add_column("Metric", style="bold")
        table.add_column("Value", style="cyan")

        table.add_row("Scanned", str(summary.scanned))
        table.add_row("Ingested", str(summary.ingested))
        table.add_row("Skipped", str(summary.skipped))
        table.add_row("Marked deleted", str(summary.marked_deleted))
        table.add_row("Errors", str(summary.errors))
        table.add_row("Duration", f"{summary.duration_seconds:.1f}s")

        console.print(table)
    else:
        print(f"  Scanned: {summary.scanned}")
        print(f"  Ingested: {summary.ingested}")
        print(f"  Skipped: {summary.skipped}")
        print(f"  Marked deleted: {summary.marked_deleted}")
        print(f"  Errors: {summary.errors}")
        print(f"  Duration: {summary.duration_seconds:.1f}s")

    if summary.errors > 0:
        print()
        ui.warning(f"{summary.errors} errors occurred:")
        for err in summary.error_details[:5]:
            ui.info(f"  • {err}")
        if len(summary.error_details) > 5:
            ui.info(f"  ... and {len(summary.error_details) - 5} more")

    print()
    ui.success(f"Documents ingested into collection '{collection}'")
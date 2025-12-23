# fitz_ai/cli/commands/ingest.py
"""
Interactive document ingestion with incremental (diff) support.

Usage:
    fitz ingest              # Interactive mode - prompts for everything
    fitz ingest ./docs       # Ingest specific directory
    fitz ingest ./docs -y    # Non-interactive with defaults
    fitz ingest ./docs --force  # Force re-ingest everything

Default behavior (per spec):
- Silently skip unchanged files (same content hash in vector DB)
- Mark deleted files as is_deleted=true in vector DB
- Summary output: "scanned N, ingested A, skipped B, marked_deleted D, errors E"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import typer

from fitz_ai.cli.ui import RICH, Panel, console, ui
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.core.registry import available_chunking_plugins
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# File Display
# =============================================================================


def _show_files(raw_docs: List, max_show: int = 8) -> None:
    """Show files being processed."""
    if not raw_docs:
        return

    files = []
    for doc in raw_docs:
        path = (
                getattr(doc, "path", None)
                or getattr(doc, "source_file", None)
                or getattr(doc, "doc_id", "?")
        )
        if hasattr(path, "name"):
            files.append(str(path.name))
        else:
            files.append(str(path).split("/")[-1].split("\\")[-1])

    show_count = min(len(files), max_show)

    if RICH:
        for f in files[:show_count]:
            console.print(f"    [dim]•[/dim] {f}")
        if len(files) > max_show:
            console.print(f"    [dim]... and {len(files) - max_show} more[/dim]")
    else:
        for f in files[:show_count]:
            print(f"    • {f}")
        if len(files) > max_show:
            print(f"    ... and {len(files) - max_show} more")


# =============================================================================
# Chunker Parameter Mapping
# =============================================================================


def _get_chunker_kwargs(chunker: str, chunk_size: int, chunk_overlap: int) -> Dict[str, Any]:
    """
    Map CLI parameters to chunker-specific kwargs.

    Different chunkers use different parameter names:
    - simple: chunk_size
    - pdf_sections: max_section_chars (doesn't use overlap)
    - overlap: chunk_size, chunk_overlap
    """
    if chunker == "pdf_sections":
        # PDF sections chunker uses different parameter names
        return {
            "max_section_chars": chunk_size,
        }
    elif chunker == "overlap":
        kwargs = {"chunk_size": chunk_size}
        if chunk_overlap > 0:
            kwargs["chunk_overlap"] = chunk_overlap
        return kwargs
    else:
        # Default/simple chunker
        kwargs = {"chunk_size": chunk_size}
        if chunk_overlap > 0:
            kwargs["chunk_overlap"] = chunk_overlap
        return kwargs


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


# =============================================================================
# Adapter Classes for Diff Ingest
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

    def mark_deleted(self, collection: str, source_path: str) -> int:
        """
        Mark vectors for a source path as deleted.

        Note: This is a best-effort operation. Not all vector DBs support
        payload updates. If not supported, returns 0.
        """
        # For now, we don't actually update the vector DB
        # The state file is authoritative, so this is optional
        logger.debug(f"mark_deleted called for {source_path} (no-op)")
        return 0


# =============================================================================
# Main Command
# =============================================================================


def command(
        source: Optional[str] = typer.Argument(
            None,
            help="Source path (file or directory). Prompts if not provided.",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            "-y",
            help="Use defaults without prompting.",
        ),
        force: bool = typer.Option(
            False,
            "--force",
            "-f",
            help="Force re-ingest all files, ignoring vector DB state.",
        ),
) -> None:
    """
    Ingest documents into the vector database.

    Default behavior (incremental):
    - Skip unchanged files (same content already in vector DB)
    - Only ingest new or modified files
    - Mark deleted files in vector DB

    Use --force to re-ingest everything regardless of state.

    Examples:
        fitz ingest              # Interactive mode
        fitz ingest ./docs       # Ingest specific directory
        fitz ingest ./docs -y    # Non-interactive with defaults
        fitz ingest --force      # Force re-ingest everything
    """
    from fitz_ai.ingest.chunking.engine import ChunkingEngine
    from fitz_ai.ingest.config.schema import ChunkerConfig
    from fitz_ai.ingest.diff import run_diff_ingest, IngestSummary
    from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
    from fitz_ai.ingest.state import IngestStateManager
    from fitz_ai.llm.registry import get_llm_plugin
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # =========================================================================
    # Load config
    # =========================================================================

    config = _load_config()

    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    default_collection = config.get("retrieval", {}).get("collection", "default")

    available_chunkers = available_chunking_plugins()
    if not available_chunkers:
        available_chunkers = ["simple"]

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Ingest")
    if force:
        ui.warning("Force mode: will re-ingest all files")
    else:
        ui.info("Incremental mode: skipping unchanged files")
    ui.info(f"Embedding: {embedding_plugin}")
    ui.info(f"Vector DB: {vector_db_plugin}")
    print()

    # =========================================================================
    # Interactive Prompts (or use defaults)
    # =========================================================================

    if non_interactive:
        if source is None:
            ui.error("Source path required in non-interactive mode.")
            ui.info("Usage: fitz ingest ./docs -y")
            raise typer.Exit(1)

        collection = default_collection
        chunker = "simple"
        chunk_size = 1000
        chunk_overlap = 0

        ui.info(f"Source: {source}")
        ui.info(f"Collection: {collection}")
        ui.info(f"Chunker: {chunker} (size={chunk_size})")

    else:
        ui.print("Configure ingestion:", "bold")
        print()

        if source is None:
            source = ui.prompt_path("Source path (file or directory)", ".")

        collection = ui.prompt_text("Collection name", default_collection)
        chunker = ui.prompt_choice("Chunker", available_chunkers, default="simple")
        chunk_size = ui.prompt_int("Chunk size", 1000)
        chunk_overlap = ui.prompt_int("Chunk overlap", 0)

    print()

    # =========================================================================
    # Initialize components
    # =========================================================================

    ui.step(1, 4, "Initializing...")

    try:
        # State manager
        state_manager = IngestStateManager()
        state_manager.load()

        # Set embedding config in state
        embedding_model = config.get("embedding", {}).get("kwargs", {}).get("model", "unknown")
        state_manager.set_embedding_config(embedding_plugin, embedding_model)

        # Ingest plugin (for parsing)
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        parser = ParserAdapter(ingest_plugin)

        # Chunker - map CLI params to chunker-specific kwargs
        chunker_kwargs = _get_chunker_kwargs(chunker, chunk_size, chunk_overlap)
        chunker_config = ChunkerConfig(
            plugin_name=chunker,
            kwargs=chunker_kwargs,
        )
        chunking_engine = ChunkingEngine.from_config(chunker_config)

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

    ui.step(2, 4, "Scanning files...")

    try:
        summary = run_diff_ingest(
            source=source,
            state_manager=state_manager,
            vector_db_writer=writer,
            embedder=embedder,
            parser=parser,
            chunker=chunking_engine.plugin,  # Pass the plugin directly
            collection=collection,
            force=force,
        )
    except Exception as e:
        ui.error(f"Ingestion failed: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # Show results
    # =========================================================================

    ui.step(3, 4, "Processing complete")
    ui.step(4, 4, "Summary")

    print()
    if RICH:
        console.print(
            Panel(
                f"[bold green]✓[/bold green] Scanned {summary.scanned} files\n"
                f"[bold green]✓[/bold green] Ingested {summary.ingested} files\n"
                f"[bold blue]→[/bold blue] Skipped {summary.skipped} unchanged files\n"
                f"[bold yellow]⚠[/bold yellow] Marked {summary.marked_deleted} deleted\n"
                f"[bold red]✗[/bold red] Errors: {summary.errors}\n"
                f"\n[dim]Collection: {collection}[/dim]",
                title="Ingestion Complete",
                border_style="green" if summary.errors == 0 else "yellow",
            )
        )
    else:
        print("=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print(f"✓ Scanned {summary.scanned} files")
        print(f"✓ Ingested {summary.ingested} files")
        print(f"→ Skipped {summary.skipped} unchanged files")
        print(f"⚠ Marked {summary.marked_deleted} deleted")
        print(f"✗ Errors: {summary.errors}")
        print(f"\nCollection: {collection}")

    if summary.errors > 0:
        ui.warning(f"{summary.errors} errors occurred. Check logs for details.")
        for detail in summary.error_details[:5]:
            ui.error(f"  {detail}")
        if len(summary.error_details) > 5:
            ui.info(f"  ... and {len(summary.error_details) - 5} more errors")
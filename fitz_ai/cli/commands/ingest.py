# fitz_ai/cli/commands/ingest.py
"""
Interactive document ingestion.

Usage:
    fitz ingest              # Interactive mode - prompts for everything
    fitz ingest ./docs       # Skip source prompt
    fitz ingest -y           # Use defaults, no prompts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import typer

from fitz_ai.cli.ui import RICH, Panel, console, ui
from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.core.registry import available_chunking_plugins
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.engine import ChunkingEngine
from fitz_ai.ingest.config.schema import ChunkerConfig
from fitz_ai.ingest.ingestion.engine import IngestionEngine
from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.vector_db.registry import get_vector_db_plugin
from fitz_ai.vector_db.writer import VectorDBWriter

logger = get_logger(__name__)


# =============================================================================
# File Display
# =============================================================================


def _show_files(raw_docs: List, max_show: int = 8) -> None:
    """Show files being processed."""
    if not raw_docs:
        return

    # Extract file paths/names
    files = []
    for doc in raw_docs:
        path = (
            getattr(doc, "path", None)
            or getattr(doc, "source_file", None)
            or getattr(doc, "doc_id", "?")
        )
        if hasattr(path, "name"):
            # Path object - get just the filename
            files.append(str(path.name))
        else:
            # String - get last part
            files.append(str(path).split("/")[-1].split("\\")[-1])

    # Show files
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
# Config Loading
# =============================================================================


def _load_config() -> dict:
    """Load config or exit with helpful message."""
    try:
        return load_config_dict(FitzPaths.config())
    except ConfigNotFoundError:
        ui.error("No config found. Run 'fitz init' first.")
        ui.info("Run: fitz init")
        raise typer.Exit(1)


# =============================================================================
# Batch Embedding with Progress
# =============================================================================


def _has_batch_embed(embedder: Any) -> bool:
    """Check if embedder supports batch embedding."""
    return hasattr(embedder, "embed_batch") or hasattr(embedder, "embed_texts")


def _embed_batch_native(embedder: Any, texts: List[str]) -> List[List[float]]:
    """Call native batch embed if available."""
    if hasattr(embedder, "embed_batch"):
        return embedder.embed_batch(texts)
    elif hasattr(embedder, "embed_texts"):
        return embedder.embed_texts(texts)
    else:
        raise NotImplementedError("No batch embed method")


def _is_batch_size_error(error: Exception) -> bool:
    """Check if error is due to batch size being too large."""
    error_str = str(error).lower()
    indicators = [
        "too many",
        "batch size",
        "maximum",
        "limit",
        "exceeded",
        "too large",
        "rate limit",
        "413",  # Payload too large
        "400",  # Bad request (often batch size)
    ]
    return any(ind in error_str for ind in indicators)


def _embed_chunks_sequential(
    embedder: Any,
    chunks: List[Chunk],
    show_progress: bool = True,
) -> List[List[float]]:
    """Embed chunks one at a time (fallback when batch not available)."""
    vectors: List[List[float]] = []
    total_chunks = len(chunks)

    with ui.progress("Embedding", total=total_chunks) as update:
        for chunk in chunks:
            vectors.append(embedder.embed(chunk.content))
            update(1)

    return vectors


def _embed_chunks(
    embedder: Any,
    chunks: List[Chunk],
    batch_size: int = 96,
    show_progress: bool = True,
) -> List[List[float]]:
    """
    Embed chunks with adaptive batching and progress bar.

    Adaptive batch sizing:
    - Starts at batch_size (default 96)
    - If batch fails, halves batch size and retries
    - Continues halving until batch_size=1 (single embedding)
    - Remembers working batch size for remaining chunks
    """
    vectors: List[List[float]] = []
    total_chunks = len(chunks)

    # Check if embedder has native batch support
    has_native_batch = _has_batch_embed(embedder)

    if not has_native_batch:
        # Fall back to single embedding
        return _embed_chunks_sequential(embedder, chunks, show_progress)

    # Adaptive batch embedding
    current_batch_size = batch_size
    position = 0

    with ui.progress(
        f"Embedding (batch={current_batch_size})", total=total_chunks
    ) as update:
        while position < total_chunks:
            batch_end = min(position + current_batch_size, total_chunks)
            batch_texts = [c.content for c in chunks[position:batch_end]]

            try:
                batch_vectors = _embed_batch_native(embedder, batch_texts)
                vectors.extend(batch_vectors)
                position = batch_end
                update(len(batch_texts))

            except Exception as e:
                if _is_batch_size_error(e) and current_batch_size > 1:
                    # Halve batch size and retry
                    current_batch_size = max(1, current_batch_size // 2)
                    if show_progress:
                        ui.warning(f"Reducing batch size to {current_batch_size}")
                else:
                    raise

    return vectors


# =============================================================================
# Chunker Parameter Mapping
# =============================================================================


def _get_chunker_kwargs(chunker: str, chunk_size: int, chunk_overlap: int) -> dict:
    """
    Map generic CLI parameters to chunker-specific kwargs.

    Different chunkers use different parameter names:
    - simple: chunk_size
    - pdf_sections: max_section_chars (doesn't use overlap)
    - overlap: chunk_size, chunk_overlap
    """
    # Map chunker names to their parameter schemas
    if chunker == "pdf_sections":
        # PDF sections chunker uses different parameter names
        kwargs = {
            "max_section_chars": chunk_size,  # Map chunk_size to max_section_chars
            # pdf_sections doesn't use overlap - it splits by sections
        }
    elif chunker == "overlap":
        # Overlap chunker uses both chunk_size and chunk_overlap
        kwargs = {"chunk_size": chunk_size}
        if chunk_overlap > 0:
            kwargs["chunk_overlap"] = chunk_overlap
    else:
        # Default/simple chunker
        kwargs = {"chunk_size": chunk_size}
        if chunk_overlap > 0:
            kwargs["chunk_overlap"] = chunk_overlap

    return kwargs


# =============================================================================
# Main Command
# =============================================================================


def command(
    source: Optional[Path] = typer.Argument(
        None,
        help="Source file or directory (will prompt if not provided).",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        "-y",
        help="Use defaults without prompting.",
    ),
) -> None:
    """
    Ingest documents into the vector database.

    Run without arguments for interactive mode:
        fitz ingest

    Or provide source directly:
        fitz ingest ./docs

    Use -y for non-interactive mode with defaults:
        fitz ingest ./docs -y
    """
    # =========================================================================
    # Load config (for defaults)
    # =========================================================================

    config = _load_config()

    # Extract defaults from config
    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    default_collection = config.get("retriever", {}).get("collection", "default")

    # Get available chunkers
    available_chunkers = available_chunking_plugins()
    if not available_chunkers:
        available_chunkers = ["simple"]

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Ingest")
    ui.info(f"Embedding: {embedding_plugin}")
    ui.info(f"Vector DB: {vector_db_plugin}")
    print()

    # =========================================================================
    # Interactive Prompts (or use defaults)
    # =========================================================================

    if non_interactive:
        # Use defaults
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
        # Interactive mode
        ui.print("Configure ingestion:", "bold")
        print()

        # Source path
        if source is None:
            source = ui.prompt_path("Source path (file or directory)", ".")
        else:
            # Validate provided source
            if not source.exists():
                ui.error(f"Source does not exist: {source}")
                raise typer.Exit(1)
            ui.info(f"Source: {source}")

        # Collection name
        collection = ui.prompt_text("Collection name", default_collection)

        # Chunking strategy
        chunker = ui.prompt_choice("Chunking strategy", available_chunkers, "simple")

        # Chunk size (with context-appropriate prompt based on chunker)
        if chunker == "pdf_sections":
            chunk_size = ui.prompt_int(
                "Max section size (characters)",
                3000,  # Default for pdf_sections
            )
        else:
            chunk_size = ui.prompt_int("Chunk size (characters)", 1000)

        # Chunk overlap (only for chunkers that support it)
        if chunker == "pdf_sections":
            chunk_overlap = 0  # PDF sections doesn't use overlap
        else:
            chunk_overlap = ui.prompt_int("Chunk overlap", 0)

        print()

    # =========================================================================
    # Confirm before proceeding
    # =========================================================================

    if not non_interactive:
        if RICH:
            console.print(
                Panel(
                    f"[bold]Source:[/bold] {source}\n"
                    f"[bold]Collection:[/bold] {collection}\n"
                    f"[bold]Chunker:[/bold] {chunker} (size={chunk_size}, overlap={chunk_overlap})\n"
                    f"[bold]Embedding:[/bold] {embedding_plugin}\n"
                    f"[bold]Vector DB:[/bold] {vector_db_plugin}",
                    title="Summary",
                    border_style="green",
                )
            )
        else:
            print("\nSummary:")
            print(f"  Source: {source}")
            print(f"  Collection: {collection}")
            print(f"  Chunker: {chunker} (size={chunk_size}, overlap={chunk_overlap})")
            print(f"  Embedding: {embedding_plugin}")
            print(f"  Vector DB: {vector_db_plugin}")

        if not ui.prompt_confirm("Proceed with ingestion?", default=True):
            ui.warning("Cancelled.")
            raise typer.Exit(0)

        print()

    # =========================================================================
    # Step 1: Read documents
    # =========================================================================

    ui.step(1, 4, "Reading documents...")

    try:
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin, kwargs={})
        raw_docs = list(ingest_engine.run(str(source)))
    except Exception as e:
        ui.error(f"Failed to read documents: {e}")
        raise typer.Exit(1)

    if not raw_docs:
        ui.error("No documents found.")
        raise typer.Exit(1)

    # Show files being processed
    _show_files(raw_docs, max_show=8)

    ui.success(f"Found {len(raw_docs)} documents")

    # =========================================================================
    # Step 2: Chunk documents
    # =========================================================================

    ui.step(2, 4, f"Chunking ({chunker})...")

    # FIX: Map CLI parameters to chunker-specific kwargs
    chunker_kwargs = _get_chunker_kwargs(chunker, chunk_size, chunk_overlap)

    chunker_config = ChunkerConfig(plugin_name=chunker, kwargs=chunker_kwargs)

    try:
        chunking_engine = ChunkingEngine.from_config(chunker_config)
    except Exception as e:
        ui.error(f"Failed to initialize chunker: {e}")
        raise typer.Exit(1)

    chunks: List[Chunk] = []
    for raw_doc in raw_docs:
        try:
            doc_chunks = chunking_engine.run(raw_doc)
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.warning(f"Failed to chunk {getattr(raw_doc, 'path', '?')}: {e}")

    if not chunks:
        ui.error("No chunks created.")
        raise typer.Exit(1)

    ui.success(f"Created {len(chunks)} chunks")

    # =========================================================================
    # Step 3: Embed chunks
    # =========================================================================

    ui.step(3, 4, f"Embedding ({embedding_plugin})...")

    try:
        # FIX: get_llm_plugin returns an INSTANCE, not a class
        embedder = get_llm_plugin(plugin_type="embedding", plugin_name=embedding_plugin)
    except Exception as e:
        ui.error(f"Failed to initialize embedder: {e}")
        raise typer.Exit(1)

    # Use batch size of 96 (Cohere max) for efficiency
    embed_batch_size = 96

    try:
        # FIX: Pass embedder instance to _embed_chunks
        vectors = _embed_chunks(embedder, chunks, batch_size=embed_batch_size)
    except Exception as e:
        ui.error(f"Failed to embed chunks: {e}")
        raise typer.Exit(1)

    ui.success(f"Generated {len(vectors)} embeddings")

    # =========================================================================
    # Step 4: Store in vector DB
    # =========================================================================

    ui.step(4, 4, "Storing in vector database...")

    try:
        # FIX: get_vector_db_plugin returns an INSTANCE, not a class
        vector_client = get_vector_db_plugin(vector_db_plugin)

        # Write to vector DB
        writer = VectorDBWriter(client=vector_client)
        writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

        ui.success(f"Stored {len(chunks)} chunks in collection '{collection}'")
    except Exception as e:
        ui.error(f"Failed to store in vector DB: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # Success!
    # =========================================================================

    print()
    if RICH:
        console.print(
            Panel(
                f"[bold green]✓[/bold green] Successfully ingested {len(raw_docs)} documents\n"
                f"[bold green]✓[/bold green] Created {len(chunks)} chunks\n"
                f"[bold green]✓[/bold green] Stored in collection: [cyan]{collection}[/cyan]",
                title="Ingestion Complete",
                border_style="green",
            )
        )
    else:
        print("=" * 60)
        print("Ingestion Complete!")
        print("=" * 60)
        print(f"✓ Ingested {len(raw_docs)} documents")
        print(f"✓ Created {len(chunks)} chunks")
        print(f"✓ Stored in collection: {collection}")
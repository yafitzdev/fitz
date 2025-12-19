# fitz_ai/ingest/cli/run.py
"""
Run command: Ingest documents into vector database.

Usage:
    fitz ingest ./documents collection
    fitz ingest ./documents collection --chunker pdf_sections
    fitz ingest ./documents collection --embedding openai
    fitz ingest ./documents collection -q
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List

import typer

from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.engine import ChunkingEngine
from fitz_ai.ingest.config.schema import ChunkerConfig
from fitz_ai.ingest.ingestion.engine import IngestionEngine
from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CHUNKING, CLI, EMBEDDING, INGEST, VECTOR_DB
from fitz_ai.vector_db.registry import get_vector_db_plugin
from fitz_ai.vector_db.writer import VectorDBWriter

logger = get_logger(__name__)

# Try to import rich for progress bars
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def _embed_with_progress(
        embed_plugin: Any,
        chunks: List[Chunk],
        show_progress: bool = True,
) -> List[List[float]]:
    """
    Embed chunks with progress bar.

    Args:
        embed_plugin: YAML embedding plugin instance with embed() method
        chunks: List of chunks to embed
        show_progress: Whether to show progress bar

    Returns:
        List of embedding vectors
    """
    vectors = []

    if RICH_AVAILABLE and show_progress and len(chunks) > 1:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=len(chunks))

            for chunk in chunks:
                vector = embed_plugin.embed(chunk.content)
                vectors.append(vector)
                progress.update(task, advance=1)
    else:
        # Fallback without rich
        for i, chunk in enumerate(chunks):
            vector = embed_plugin.embed(chunk.content)
            vectors.append(vector)
            if not RICH_AVAILABLE and (i + 1) % 10 == 0:
                typer.echo(f"  Embedded {i + 1}/{len(chunks)} chunks...")

    return vectors


def command(
        source: Path = typer.Argument(
            ...,
            help="Source to ingest (file or directory).",
        ),
        collection: str = typer.Argument(
            "default",
            help="Target collection name.",
        ),
        ingest_plugin: str = typer.Option(
            "local",
            "--ingest",
            "-i",
            help="Ingestion plugin name.",
        ),
        chunker: str = typer.Option(
            "simple",
            "--chunker",
            "-c",
            help="Chunking plugin to use.",
        ),
        chunk_size: int = typer.Option(
            1000,
            "--chunk-size",
            help="Target chunk size in characters.",
        ),
        chunk_overlap: int = typer.Option(
            0,
            "--chunk-overlap",
            help="Overlap between chunks in characters.",
        ),
        min_section_chars: int = typer.Option(
            50,
            "--min-section-chars",
            help="Minimum section size for section-based chunkers.",
        ),
        max_section_chars: int = typer.Option(
            3000,
            "--max-section-chars",
            help="Maximum section size for section-based chunkers.",
        ),
        embedding_plugin: str = typer.Option(
            "cohere",
            "--embedding",
            "-e",
            help="Embedding plugin name.",
        ),
        vector_db_plugin: str = typer.Option(
            "qdrant",
            "--vector-db",
            "-v",
            help="Vector DB plugin name.",
        ),
        batch_size: int = typer.Option(
            50,
            "--batch-size",
            "-b",
            help="Batch size for vector DB writes.",
        ),
        quiet: bool = typer.Option(
            False,
            "--quiet",
            "-q",
            help="Suppress progress output.",
        ),
) -> None:
    """
    Ingest documents into a vector database.

    Examples:
        fitz ingest ./docs default
        fitz ingest ./docs my_knowledge
        fitz ingest ./docs my_docs --embedding openai
        fitz ingest ./docs my_docs -q
        fitz ingest ./doc.pdf papers --chunker pdf_sections
        fitz ingest ./doc.pdf papers --chunker simple --chunk-size 500
    """
    # Validate source path
    if not source.exists():
        typer.echo(f"ERROR: source does not exist: {source}")
        raise typer.Exit(code=1)

    if not (source.is_file() or source.is_dir()):
        typer.echo(f"ERROR: source is not file or directory: {source}")
        raise typer.Exit(code=1)

    logger.info(
        f"{CLI}{INGEST} Starting ingestion: source='{source}' → collection='{collection}' "
        f"(ingest='{ingest_plugin}', chunker='{chunker}', embedding='{embedding_plugin}', vdb='{vector_db_plugin}')"
    )

    show_progress = not quiet

    # Header
    if show_progress:
        if RICH_AVAILABLE:
            console.print(
                Panel.fit(
                    f"[bold]Ingesting[/bold] {source}\n"
                    f"[dim]Collection: {collection}[/dim]\n"
                    f"[dim]Chunker: {chunker}[/dim]",
                    title="🔄 fitz ingest",
                    border_style="blue",
                )
            )
        else:
            typer.echo()
            typer.echo("=" * 60)
            typer.echo(f"Ingesting: {source}")
            typer.echo(f"Collection: {collection}")
            typer.echo(f"Chunker: {chunker}")
            typer.echo("=" * 60)

    # =========================================================================
    # Step 1: Ingest documents
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            with console.status("[bold blue]Reading documents...", spinner="dots"):
                IngestPluginCls = get_ingest_plugin(ingest_plugin)
                ingest_plugin_obj = IngestPluginCls()
                ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
                raw_docs = list(ingest_engine.run(str(source)))
        else:
            typer.echo("[1/4] Reading documents...")
            IngestPluginCls = get_ingest_plugin(ingest_plugin)
            ingest_plugin_obj = IngestPluginCls()
            ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
            raw_docs = list(ingest_engine.run(str(source)))
    else:
        IngestPluginCls = get_ingest_plugin(ingest_plugin)
        ingest_plugin_obj = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
        raw_docs = list(ingest_engine.run(str(source)))

    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents from {source}")

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Found [bold]{len(raw_docs)}[/bold] documents")
        else:
            typer.echo(f"  ✓ Found {len(raw_docs)} documents")

    if not raw_docs:
        typer.echo("No documents found to ingest.")
        raise typer.Exit(code=0)

    # =========================================================================
    # Step 2: Chunk documents with configured strategy
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[bold blue]⏳[/bold blue] Chunking with {chunker}...")
        else:
            typer.echo(f"[2/4] Chunking with {chunker}...")

    # Build chunker kwargs dynamically based on CLI parameters AND chunker type
    chunker_kwargs: Dict[str, Any] = {}

    # Section-based chunkers use different parameters
    if "section" in chunker.lower():
        # Section chunkers: min_section_chars, max_section_chars, preserve_short_sections
        if min_section_chars != 50:
            chunker_kwargs["min_section_chars"] = min_section_chars
        if max_section_chars != 3000:
            chunker_kwargs["max_section_chars"] = max_section_chars
        # Always include preserve_short_sections for section chunkers
        chunker_kwargs["preserve_short_sections"] = True
    else:
        # Standard chunkers: chunk_size, chunk_overlap
        if chunk_size != 1000:
            chunker_kwargs["chunk_size"] = chunk_size
        if chunk_overlap > 0:
            chunker_kwargs["chunk_overlap"] = chunk_overlap

    # Create chunker config and engine
    chunker_config = ChunkerConfig(plugin_name=chunker, kwargs=chunker_kwargs)

    try:
        chunking_engine = ChunkingEngine.from_config(chunker_config)
    except Exception as e:
        typer.echo(f"ERROR: Failed to initialize chunker '{chunker}': {e}")
        typer.echo()
        typer.echo("Available chunkers:")
        from fitz_ai.core.registry import available_chunking_plugins
        for name in available_chunking_plugins():
            typer.echo(f"  • {name}")
        raise typer.Exit(code=1)

    # Process all documents through chunking engine
    chunks: List[Chunk] = []
    for raw_doc in raw_docs:
        try:
            doc_chunks = chunking_engine.run(raw_doc)
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.error(f"{CHUNKING} Failed to chunk document: {e}")
            typer.echo(f"WARNING: Failed to chunk {getattr(raw_doc, 'path', 'unknown')}: {e}")

    if not chunks:
        typer.echo("ERROR: No chunks were created. Check your documents and chunker settings.")
        raise typer.Exit(code=1)

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks")
        else:
            typer.echo(f"  ✓ Created {len(chunks)} chunks")

    logger.info(f"{CHUNKING} Created {len(chunks)} chunks using '{chunker}' chunker")

    # =========================================================================
    # Step 3: Generate embeddings
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print(
                f"[bold blue]⏳[/bold blue] Generating embeddings with {embedding_plugin}..."
            )
        else:
            typer.echo(f"[3/4] Generating embeddings with {embedding_plugin}...")

    # Use YAML plugin directly - no EmbeddingEngine wrapper needed
    embed_plugin = get_llm_plugin(plugin_type="embedding", plugin_name=embedding_plugin)

    vectors = _embed_with_progress(embed_plugin, chunks, show_progress=show_progress)

    logger.info(f"{EMBEDDING} Generated {len(vectors)} embeddings")

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Generated [bold]{len(vectors)}[/bold] embeddings")
        else:
            typer.echo(f"  ✓ Generated {len(vectors)} embeddings")

    # =========================================================================
    # Step 4: Write to vector DB
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[bold blue]⏳[/bold blue] Writing to {vector_db_plugin}...")
        else:
            typer.echo(f"[4/4] Writing to {vector_db_plugin}...")

    vdb_plugin = get_vector_db_plugin(vector_db_plugin)

    writer = VectorDBWriter(client=vdb_plugin)

    # Batch write
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i: i + batch_size]
        batch_vectors = vectors[i: i + batch_size]
        writer.upsert(collection=collection, chunks=batch_chunks, vectors=batch_vectors)
        logger.debug(f"{VECTOR_DB} Wrote batch {i // batch_size + 1}")

    logger.info(f"{VECTOR_DB} Written {len(chunks)} chunks to collection '{collection}'")

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Written to collection [bold]{collection}[/bold]")
        else:
            typer.echo(f"  ✓ Written to collection {collection}")

    # =========================================================================
    # Summary
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print()
            console.print("[bold green]✅ Ingestion Complete[/bold green]")

            table = Table(show_header=False, box=None)
            table.add_column("Key", style="dim")
            table.add_column("Value", style="bold")
            table.add_row("Documents", str(len(raw_docs)))
            table.add_row("Chunks", str(len(chunks)))
            table.add_row("Chunker", chunker)
            table.add_row("Collection", collection)
            table.add_row("Vector DB", vector_db_plugin)
            table.add_row("Embedding", embedding_plugin)
            console.print(table)

            console.print(f'\n[dim]Next:[/dim] fitz query "Your question" --collection {collection}')
        else:
            typer.echo()
            typer.echo("✅ Ingestion Complete")
            typer.echo(f"  Documents: {len(raw_docs)}")
            typer.echo(f"  Chunks: {len(chunks)}")
            typer.echo(f"  Chunker: {chunker}")
            typer.echo(f"  Collection: {collection}")
            typer.echo(f"  Vector DB: {vector_db_plugin}")
            typer.echo(f"  Embedding: {embedding_plugin}")
            typer.echo()
            typer.echo(f'Next: fitz query "Your question" --collection {collection}')
# fitz/ingest/cli/run.py
"""
Run command: Ingest documents into vector database.

Usage:
    fitz ingest ./documents collection
    fitz ingest ./documents collection --embedding openai
    fitz ingest ./documents collection -q
"""

from pathlib import Path
from typing import Any, Iterable, List

import typer

from fitz.engines.classic_rag.models.chunk import Chunk
from fitz.ingest.ingestion.engine import IngestionEngine
from fitz.ingest.ingestion.registry import get_ingest_plugin
from fitz.llm.registry import get_llm_plugin
from fitz.logging.logger import get_logger
from fitz.logging.tags import CHUNKING, CLI, EMBEDDING, INGEST, VECTOR_DB
from fitz.vector_db.registry import get_vector_db_plugin
from fitz.vector_db.writer import VectorDBWriter

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


def _raw_to_chunks(raw_docs: Iterable[Any]) -> list[Chunk]:
    """Convert raw documents to canonical chunks."""
    chunks: list[Chunk] = []

    for i, doc in enumerate(raw_docs):
        meta = dict(getattr(doc, "metadata", None) or {})
        path = getattr(doc, "path", None)
        if path:
            meta.setdefault("path", str(path))

        # Be robust across RawDocument variants: some use .content, some .text
        content = getattr(doc, "content", None)
        if content is None:
            content = getattr(doc, "text", None)
        content = content or ""

        chunks.append(
            Chunk(
                id=f"{meta.get('path', 'doc')}:{i}",
                doc_id=str(meta.get("path") or meta.get("doc_id") or "unknown"),
                chunk_index=i,
                content=content,
                metadata=meta,
            )
        )

    return chunks


def _embed_with_progress(
    embed_plugin: Any,
    chunks: List[Chunk],
    show_progress: bool = True,
) -> List[List[float]]:
    """Embed chunks with progress bar.

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
        f"(ingest='{ingest_plugin}', embedding='{embedding_plugin}', vdb='{vector_db_plugin}')"
    )

    show_progress = not quiet

    # Header
    if show_progress:
        if RICH_AVAILABLE:
            console.print(
                Panel.fit(
                    f"[bold]Ingesting[/bold] {source}\n" f"[dim]Collection: {collection}[/dim]",
                    title="🔄 fitz ingest",
                    border_style="blue",
                )
            )
        else:
            typer.echo()
            typer.echo("=" * 60)
            typer.echo(f"Ingesting: {source}")
            typer.echo(f"Collection: {collection}")
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
                raw_docs = list(ingest_engine.ingest(str(source)))
        else:
            typer.echo("[1/4] Reading documents...")
            IngestPluginCls = get_ingest_plugin(ingest_plugin)
            ingest_plugin_obj = IngestPluginCls()
            ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
            raw_docs = list(ingest_engine.ingest(str(source)))
    else:
        IngestPluginCls = get_ingest_plugin(ingest_plugin)
        ingest_plugin_obj = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
        raw_docs = list(ingest_engine.ingest(str(source)))

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
    # Step 2: Convert to chunks
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print("[bold blue]⏳[/bold blue] Creating chunks...")
        else:
            typer.echo("[2/4] Creating chunks...")

    chunks = _raw_to_chunks(raw_docs)

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks")
        else:
            typer.echo(f"  ✓ Created {len(chunks)} chunks")

    logger.info(f"{CHUNKING} Created {len(chunks)} chunks from raw documents")

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

    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vdb_plugin = VectorDBPluginCls()

    writer = VectorDBWriter(client=vdb_plugin)

    # Batch write
    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i : i + batch_size]
        batch_vectors = vectors[i : i + batch_size]
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
            table.add_row("Collection", collection)
            table.add_row("Vector DB", vector_db_plugin)
            table.add_row("Embedding", embedding_plugin)
            console.print(table)

            console.print('\n[dim]Next:[/dim] fitz query "Your question"')
        else:
            typer.echo()
            typer.echo("✅ Ingestion Complete")
            typer.echo(f"  Documents: {len(raw_docs)}")
            typer.echo(f"  Chunks: {len(chunks)}")
            typer.echo(f"  Collection: {collection}")
            typer.echo(f"  Vector DB: {vector_db_plugin}")
            typer.echo(f"  Embedding: {embedding_plugin}")
            typer.echo()
            typer.echo('Next: fitz query "Your question"')

# fitz/ingest/cli/run.py
"""
Run command: Ingest documents into vector database.

Usage:
    fitz-ingest run ./documents --collection my_docs
    fitz-ingest run ./documents --collection my_docs --ingest-plugin local
    fitz-ingest run ./documents --collection my_docs --embedding-plugin cohere
"""

from pathlib import Path
from typing import Any, Iterable, List

import typer

from fitz.llm.embedding.engine import EmbeddingEngine
from fitz.llm.registry import get_llm_plugin
from fitz.logging.logger import get_logger
from fitz.logging.tags import CHUNKING, CLI, EMBEDDING, INGEST, VECTOR_DB
from fitz.engines.classic_rag.models.chunk import Chunk
from fitz.vector_db.registry import get_vector_db_plugin
from fitz.vector_db.writer import VectorDBWriter
from fitz.ingest.ingestion.engine import IngestionEngine
from fitz.ingest.ingestion.registry import get_ingest_plugin

logger = get_logger(__name__)

# Try to import rich for progress bars
try:
    from rich.console import Console
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    from rich.panel import Panel
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
    embed_engine: EmbeddingEngine,
    chunks: List[Chunk],
    show_progress: bool = True,
) -> List[List[float]]:
    """Embed chunks with progress bar."""
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
                vector = embed_engine.embed(chunk.content)
                vectors.append(vector)
                progress.update(task, advance=1)
    else:
        # Fallback without rich
        for i, chunk in enumerate(chunks):
            vector = embed_engine.embed(chunk.content)
            vectors.append(vector)
            if not RICH_AVAILABLE and (i + 1) % 10 == 0:
                typer.echo(f"  Embedded {i + 1}/{len(chunks)} chunks...")
    
    return vectors


def command(
    source: Path = typer.Argument(
        ...,
        help="Source to ingest (file or directory, depending on ingest plugin).",
    ),
    collection: str = typer.Option(
        ...,
        "--collection",
        "-c",
        help="Target vector DB collection name.",
    ),
    ingest_plugin: str = typer.Option(
        "local",
        "--ingest-plugin",
        "-i",
        help="Ingestion plugin name (as registered in ingest.ingestion.registry).",
    ),
    embedding_plugin: str = typer.Option(
        "cohere",
        "--embedding-plugin",
        "-e",
        help="Embedding plugin name (registered in core.llm.registry).",
    ),
    vector_db_plugin: str = typer.Option(
        "qdrant",
        "--vector-db-plugin",
        "-v",
        help="Vector DB plugin name (registered in core.vector_db.registry).",
    ),
    batch_size: int = typer.Option(
        50,
        "--batch-size",
        "-b",
        help="Number of chunks to process before writing to vector DB.",
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

    This command performs the complete ingestion pipeline:
    1. Ingest documents from source (file or directory)
    2. Convert to chunks
    3. Generate embeddings
    4. Store in vector database

    Examples:
        # Ingest local documents with default plugins
        fitz-ingest run ./docs --collection my_knowledge

        # Use specific plugins
        fitz-ingest run ./docs --collection my_docs \\
            --ingest-plugin local \\
            --embedding-plugin openai \\
            --vector-db-plugin qdrant
        
        # Quiet mode (minimal output)
        fitz-ingest run ./docs --collection my_docs -q
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
            console.print(Panel.fit(
                f"[bold]Ingesting[/bold] {source}\n"
                f"[dim]Collection: {collection}[/dim]",
                title="🔄 fitz-ingest",
                border_style="blue"
            ))
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
                raw_docs = list(ingest_engine.run(str(source)))
            console.print(f"[green]✓[/green] Found [bold]{len(raw_docs)}[/bold] documents")
        else:
            typer.echo(f"[1/4] Ingesting documents from {source}...")
            IngestPluginCls = get_ingest_plugin(ingest_plugin)
            ingest_plugin_obj = IngestPluginCls()
            ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
            raw_docs = list(ingest_engine.run(str(source)))
            typer.echo(f"  ✓ Ingested {len(raw_docs)} documents")
    else:
        IngestPluginCls = get_ingest_plugin(ingest_plugin)
        ingest_plugin_obj = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin_obj, kwargs={})
        raw_docs = list(ingest_engine.run(str(source)))

    logger.info(f"{INGEST} Ingested {len(raw_docs)} raw documents")

    if not raw_docs:
        if RICH_AVAILABLE:
            console.print("[yellow]⚠[/yellow] No documents found to ingest")
        else:
            typer.echo("⚠ No documents found to ingest")
        raise typer.Exit(code=0)

    # =========================================================================
    # Step 2: Convert to chunks
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            with console.status("[bold blue]Converting to chunks...", spinner="dots"):
                chunks = _raw_to_chunks(raw_docs)
            console.print(f"[green]✓[/green] Created [bold]{len(chunks)}[/bold] chunks")
        else:
            typer.echo("[2/4] Converting to chunks...")
            chunks = _raw_to_chunks(raw_docs)
            typer.echo(f"  ✓ Created {len(chunks)} chunks")
    else:
        chunks = _raw_to_chunks(raw_docs)

    logger.info(f"{CHUNKING} Produced {len(chunks)} chunks (1:1 raw→chunk)")

    # =========================================================================
    # Step 3: Generate embeddings
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[blue]⏳[/blue] Generating embeddings with [bold]{embedding_plugin}[/bold]...")
        else:
            typer.echo(f"[3/4] Generating embeddings with '{embedding_plugin}'...")

    EmbedPluginCls = get_llm_plugin(plugin_name=embedding_plugin, plugin_type="embedding")
    embed_engine = EmbeddingEngine(EmbedPluginCls())
    
    vectors = _embed_with_progress(embed_engine, chunks, show_progress=show_progress)
    
    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Generated [bold]{len(vectors)}[/bold] embeddings")
        else:
            typer.echo(f"  ✓ Generated {len(vectors)} embeddings")

    logger.info(f"{EMBEDDING} Embedded {len(vectors)} chunks using '{embedding_plugin}'")

    # =========================================================================
    # Step 4: Write to vector database
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[blue]⏳[/blue] Writing to [bold]{vector_db_plugin}[/bold]...")
        else:
            typer.echo(f"[4/4] Writing to vector database '{vector_db_plugin}'...")

    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)

    # Special handling for local-faiss which needs dimension
    if vector_db_plugin == "local-faiss":
        if not vectors:
            typer.echo("ERROR: No vectors generated, cannot initialize FAISS")
            raise typer.Exit(code=1)

        dim = len(vectors[0])
        from fitz.backends.local_vector_db.config import LocalVectorDBConfig
        config = LocalVectorDBConfig()
        vdb_client = VectorDBPluginCls(dim=dim, config=config)
    else:
        vdb_client = VectorDBPluginCls()

    writer = VectorDBWriter(client=vdb_client)
    
    # Write in batches with progress
    if RICH_AVAILABLE and show_progress and len(chunks) > batch_size:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Writing to DB...", total=len(chunks))
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                writer.upsert(collection=collection, chunks=batch_chunks, vectors=batch_vectors)
                progress.update(task, advance=len(batch_chunks))
    else:
        writer.upsert(collection=collection, chunks=chunks, vectors=vectors)

    if show_progress:
        if RICH_AVAILABLE:
            console.print(f"[green]✓[/green] Written to collection [bold]{collection}[/bold]")
        else:
            typer.echo(f"  ✓ Written to collection '{collection}'")

    logger.info(f"{VECTOR_DB} Upserted {len(chunks)} chunks into collection='{collection}'")

    # =========================================================================
    # Summary
    # =========================================================================
    if show_progress:
        if RICH_AVAILABLE:
            console.print()
            
            # Summary table
            table = Table(title="✅ Ingestion Complete", show_header=False, box=None)
            table.add_column("Metric", style="dim")
            table.add_column("Value", style="bold")
            
            table.add_row("Documents", str(len(raw_docs)))
            table.add_row("Chunks", str(len(chunks)))
            table.add_row("Collection", collection)
            table.add_row("Vector DB", vector_db_plugin)
            table.add_row("Embedding", embedding_plugin)
            
            console.print(table)
            console.print()
            console.print("[dim]Next: fitz-pipeline query \"Your question\" [/dim]")
        else:
            typer.echo()
            typer.echo("=" * 60)
            typer.echo("✓ Ingestion complete!")
            typer.echo("=" * 60)
            typer.echo(f"Documents:  {len(raw_docs)}")
            typer.echo(f"Chunks:     {len(chunks)}")
            typer.echo(f"Collection: {collection}")
            typer.echo()
    else:
        # Quiet mode - just confirm success
        typer.echo(f"✓ Ingested {len(chunks)} chunks into '{collection}'")
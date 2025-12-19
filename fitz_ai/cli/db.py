# fitz/cli/db.py
"""
Database command: Inspect vector database collections.

Usage:
    fitz db                          # List all collections
    fitz db default                  # Show stats and samples from 'default' collection
    fitz db my_collection            # Show stats for specific collection
"""

from __future__ import annotations

from typing import Optional

import typer

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def command(
    collection: Optional[str] = typer.Argument(
        None,
        help="Collection name to inspect.",
    ),
    list_collections: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all collections.",
    ),
    samples: int = typer.Option(
        3,
        "--samples",
        "-n",
        help="Number of sample chunks to show.",
    ),
    vector_db: str = typer.Option(
        "qdrant",
        "--vector-db",
        "-v",
        help="Vector DB plugin name.",
    ),
) -> None:
    """
    Inspect vector database collections.

    Shows stats, configuration, and sample content from a collection.

    Examples:
        fitz db                       # List all collections
        fitz db default               # Inspect 'default' collection
        fitz db my_docs               # Inspect 'my_docs' collection
        fitz db default -n 5          # Show 5 sample chunks
    """
    from fitz_ai.vector_db.registry import get_vector_db_plugin

    # Get vector DB client
    VectorDBPluginCls = get_vector_db_plugin(vector_db)
    vdb = VectorDBPluginCls()

    # List mode
    if list_collections:
        _list_all_collections(vdb, vector_db)
        return

    # If no collection specified, list collections and prompt
    if collection is None:
        collections = vdb.list_collections()
        if not collections:
            typer.echo("No collections found. Run 'fitz ingest' to create one.")
            raise typer.Exit(0)

        typer.echo("Available collections:")
        for c in collections:
            typer.echo(f"  • {c}")
        typer.echo()
        typer.echo("Usage: fitz db <collection_name>")
        raise typer.Exit(0)

    # Check if collection exists
    collections = vdb.list_collections()
    if collection not in collections:
        typer.echo(f"Collection '{collection}' not found.")
        typer.echo()
        if collections:
            typer.echo("Available collections:")
            for c in collections:
                typer.echo(f"  • {c}")
        else:
            typer.echo("No collections found. Run 'fitz ingest' to create one.")
        raise typer.Exit(1)

    # Get stats
    stats = vdb.get_collection_stats(collection)

    # Display header
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                f"[bold]{collection}[/bold]",
                title="📊 Collection",
                border_style="blue",
            )
        )
    else:
        typer.echo()
        typer.echo("=" * 60)
        typer.echo(f"Collection: {collection}")
        typer.echo("=" * 60)

    # Display stats
    typer.echo()
    if RICH_AVAILABLE:
        table = Table(show_header=False, box=None)
        table.add_column("Property", style="dim")
        table.add_column("Value", style="bold")

        table.add_row(
            "Chunks",
            (
                f"{stats.get('points_count', 'N/A'):,}"
                if isinstance(stats.get("points_count"), int)
                else str(stats.get("points_count", "N/A"))
            ),
        )
        table.add_row(
            "Vectors",
            (
                f"{stats.get('vectors_count', 'N/A'):,}"
                if isinstance(stats.get("vectors_count"), int)
                else str(stats.get("vectors_count", "N/A"))
            ),
        )
        table.add_row("Status", stats.get("status", "N/A"))
        table.add_row("Vector DB", vector_db)

        console.print(table)
    else:
        typer.echo(f"  Chunks:    {stats.get('points_count', 'N/A')}")
        typer.echo(f"  Vectors:   {stats.get('vectors_count', 'N/A')}")
        typer.echo(f"  Status:    {stats.get('status', 'N/A')}")
        typer.echo(f"  Vector DB: {vector_db}")

    # Get sample chunks (Qdrant-specific for now)
    if vector_db == "qdrant" and samples > 0:
        _show_sample_chunks(vdb, collection, samples)

    typer.echo()


def _list_all_collections(vdb, vector_db: str) -> None:
    """List all collections with stats."""
    collections = vdb.list_collections()

    if not collections:
        typer.echo("No collections found.")
        typer.echo("Run 'fitz ingest ./docs collection' to create one.")
        return

    typer.echo()
    if RICH_AVAILABLE:
        table = Table(title="Vector DB Collections")
        table.add_column("Collection", style="cyan")
        table.add_column("Chunks", justify="right")
        table.add_column("Status")

        for name in sorted(collections):
            stats = vdb.get_collection_stats(name)
            points = stats.get("points_count", "?")
            if isinstance(points, int):
                points = f"{points:,}"
            status = stats.get("status", "?")
            table.add_row(name, str(points), str(status))

        console.print(table)
    else:
        typer.echo("Collections:")
        typer.echo("-" * 40)
        for name in sorted(collections):
            stats = vdb.get_collection_stats(name)
            points = stats.get("points_count", "?")
            typer.echo(f"  {name}: {points} chunks")

    typer.echo()


def _show_sample_chunks(vdb, collection: str, num_samples: int) -> None:
    """Show sample chunks from the collection."""
    typer.echo()

    try:
        # Access the underlying Qdrant client
        client = getattr(vdb, "_client", None)
        if not client:
            return

        # Scroll to get sample points
        records, _ = client.scroll(
            collection_name=collection,
            limit=num_samples,
            with_payload=True,
            with_vectors=False,
        )

        if not records:
            typer.echo("No chunks found in collection.")
            return

        if RICH_AVAILABLE:
            console.print("[bold]Sample Chunks:[/bold]")
        else:
            typer.echo("Sample Chunks:")
            typer.echo("-" * 40)

        for i, record in enumerate(records, 1):
            payload = record.payload or {}
            content = payload.get("content", payload.get("text", ""))
            doc_id = payload.get("doc_id", payload.get("source_file", "unknown"))
            chunk_index = payload.get("chunk_index", "?")

            # Truncate content for display
            if len(content) > 200:
                content = content[:200] + "..."

            if RICH_AVAILABLE:
                console.print()
                console.print(
                    f"[dim]#{i}[/dim] [cyan]{doc_id}[/cyan] [dim](chunk {chunk_index})[/dim]"
                )
                console.print(f"  [dim]{content}[/dim]")
            else:
                typer.echo()
                typer.echo(f"#{i} {doc_id} (chunk {chunk_index})")
                typer.echo(f"  {content}")

    except Exception as e:
        logger.debug(f"Could not fetch sample chunks: {e}")
        typer.echo(f"Could not fetch sample chunks: {e}")

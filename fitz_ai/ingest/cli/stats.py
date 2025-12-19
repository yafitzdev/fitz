# fitz/ingest/cli/stats.py
"""
Stats command: Show statistics about an ingested collection.

Usage:
    fitz ingest stats my_collection
    fitz ingest stats my_collection --vector-db qdrant
"""

import typer

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CLI, VECTOR_DB
from fitz_ai.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def command(
    collection: str = typer.Argument(
        ...,
        help="Collection name to show statistics for.",
    ),
    vector_db_plugin: str = typer.Option(
        "qdrant",
        "--vector-db",
        "-v",
        help="Vector DB plugin name.",
    ),
) -> None:
    """
    Show statistics about an ingested collection.

    Examples:
        fitz ingest stats default
        fitz ingest stats my_collection
        fitz ingest stats my_collection --vector-db faiss
    """
    logger.info(f"{CLI}{VECTOR_DB} Getting stats for collection: {collection}")

    # Get the vector DB plugin
    VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
    vdb = VectorDBPluginCls()

    # Check if collection exists
    collections = vdb.list_collections()
    if collection not in collections:
        typer.echo(f"Collection '{collection}' not found.")
        typer.echo(f"Available collections: {', '.join(collections) if collections else '(none)'}")
        raise typer.Exit(code=1)

    # Get stats
    stats = vdb.get_collection_stats(collection)

    typer.echo()
    if RICH_AVAILABLE:
        table = Table(title=f"Collection: {collection}")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        for key, value in stats.items():
            table.add_row(str(key), str(value))

        console.print(table)
    else:
        typer.echo(f"Collection: {collection}")
        typer.echo("-" * 40)
        for key, value in stats.items():
            typer.echo(f"  {key}: {value}")

    typer.echo()

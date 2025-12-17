"""
Stats command: Show statistics about an ingested collection.

Usage:
    fitz-ingest stats --collection my_docs
    fitz-ingest stats --collection my_docs --vector-db-plugin qdrant
"""

import typer

from fitz.logging.logger import get_logger
from fitz.logging.tags import CLI, VECTOR_DB
from fitz.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)


def command(
    collection: str = typer.Option(
        ...,
        "--collection",
        "-c",
        help="Collection name to show statistics for.",
    ),
    vector_db_plugin: str = typer.Option(
        "qdrant",
        "--vector-db-plugin",
        "-v",
        help="Vector DB plugin name.",
    ),
) -> None:
    """
    Show statistics about an ingested collection.

    Displays information about a collection in your vector database:
    - Total number of chunks/vectors
    - Collection configuration
    - Storage metrics (if available)

    Useful for:
    - Verifying successful ingestion
    - Monitoring collection size
    - Debugging retrieval issues

    Examples:
        # Show stats for a collection
        fitz-ingest stats --collection my_knowledge

        # Use specific vector DB
        fitz-ingest stats --collection my_docs --vector-db-plugin qdrant
    """
    logger.info(f"{CLI}{VECTOR_DB} Fetching stats for collection: {collection}")

    typer.echo()
    typer.echo("=" * 60)
    typer.echo(f"COLLECTION STATS: {collection}")
    typer.echo("=" * 60)
    typer.echo()

    try:
        # Get vector DB client
        VectorDBPluginCls = get_vector_db_plugin(vector_db_plugin)
        vdb_client = VectorDBPluginCls()

        # Try to get collection info
        # Note: This is plugin-dependent, so we'll handle different plugins
        typer.echo(f"Vector Database: {vector_db_plugin}")
        typer.echo()

        # For Qdrant specifically
        if vector_db_plugin == "qdrant":
            try:
                # Get Qdrant client
                client = getattr(vdb_client, "_client", None)
                if client:
                    # Get collection info
                    collection_info = client.get_collection(collection_name=collection)

                    typer.echo("Collection Information:")
                    typer.echo("-" * 40)
                    typer.echo(f"  Points count: {collection_info.points_count:,}")
                    typer.echo(f"  Vector size:  {collection_info.config.params.vectors.size}")
                    typer.echo(f"  Distance:     {collection_info.config.params.vectors.distance}")

                    if collection_info.points_count > 0:
                        typer.echo()
                        typer.echo("✓ Collection is ready for queries")
                    else:
                        typer.echo()
                        typer.echo("⚠️  Collection is empty")
                else:
                    typer.echo("⚠️  Could not access Qdrant client")
            except Exception as e:
                typer.echo(f"⚠️  Could not fetch collection info: {e}")
                typer.echo()
                typer.echo("Possible reasons:")
                typer.echo("  • Collection doesn't exist")
                typer.echo("  • Qdrant server is not running")
                typer.echo("  • Connection configuration is incorrect")
        else:
            # Generic message for other vector DBs
            typer.echo("⚠️  Stats command is currently optimized for Qdrant.")
            typer.echo(f"   Support for '{vector_db_plugin}' coming soon.")
            typer.echo()
            typer.echo("For now, use your vector DB's native tools:")
            typer.echo("  • Qdrant: http://localhost:6333/dashboard")
            typer.echo("  • Pinecone: pinecone.io dashboard")
            typer.echo("  • Weaviate: http://localhost:8080/v1/schema")

    except Exception as e:
        typer.echo(f"✗ Error: {e}")
        raise typer.Exit(code=1)

    typer.echo()
    typer.echo("=" * 60)
    typer.echo()

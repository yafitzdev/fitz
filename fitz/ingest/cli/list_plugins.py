# fitz/ingest/cli/list_plugins.py
"""
List-plugins command: Show available plugins for ingestion.

Usage:
    fitz-ingest list-plugins
    fitz-ingest list-plugins --type ingest
    fitz-ingest list-plugins --type embedding
"""

from typing import Optional

import typer

from fitz.core.registry import available_llm_plugins  # Use this for embedding/rerank
from fitz.core.registry import (
    available_ingest_plugins,
    get_ingest_plugin,
)
from fitz.vector_db.registry import available_vector_db_plugins


def command(
    type: Optional[str] = typer.Option(
        None,
        "--type",
        "-t",
        help="Filter by plugin type (ingest, embedding, rerank, vector-db, all).",
    ),
) -> None:
    """
    List all available plugins for ingestion pipeline.

    Shows plugins for:
    - Ingestion (document readers)
    - Embedding (text to vectors)
    - Reranking (result scoring)
    - Vector databases (storage)

    Examples:
        # Show all plugins
        fitz-ingest list-plugins

        # Show only ingestion plugins
        fitz-ingest list-plugins --type ingest

        # Show only embedding plugins
        fitz-ingest list-plugins --type embedding
    """
    show_all = type is None or type == "all"

    typer.echo()
    typer.echo("=" * 60)
    typer.echo("AVAILABLE PLUGINS")
    typer.echo("=" * 60)

    # Ingestion plugins
    if show_all or type == "ingest":
        typer.echo()
        typer.echo("📄 Ingestion Plugins (document readers)")
        typer.echo("-" * 60)
        ingest_plugins = available_ingest_plugins()
        if ingest_plugins:
            for name in ingest_plugins:
                cls = get_ingest_plugin(name)
                plugin_type = getattr(cls, "plugin_type", "N/A")
                doc = cls.__doc__ or "No description"
                desc = doc.strip().split("\n")[0]
                typer.echo(f"  • {name:15} [{plugin_type}]")
                typer.echo(f"    {desc}")
        else:
            typer.echo("  (No ingestion plugins found)")

    # Embedding plugins
    if show_all or type == "embedding":
        typer.echo()
        typer.echo("🔢 Embedding Plugins (text → vectors)")
        typer.echo("-" * 60)
        embedding_plugins = available_llm_plugins("embedding")
        if embedding_plugins:
            for name in embedding_plugins:
                typer.echo(f"  • {name}")
        else:
            typer.echo("  (No embedding plugins found)")

    # Rerank plugins
    if show_all or type == "rerank":
        typer.echo()
        typer.echo("🎯 Rerank Plugins (result scoring)")
        typer.echo("-" * 60)
        rerank_plugins = available_llm_plugins("rerank")
        if rerank_plugins:
            for name in rerank_plugins:
                typer.echo(f"  • {name}")
        else:
            typer.echo("  (No rerank plugins found)")

    # Vector DB plugins
    if show_all or type == "vector-db":
        typer.echo()
        typer.echo("🗄️  Vector DB Plugins (storage)")
        typer.echo("-" * 60)
        vdb_plugins = available_vector_db_plugins()
        if vdb_plugins:
            for name in vdb_plugins:
                typer.echo(f"  • {name}")
        else:
            typer.echo("  (No vector DB plugins found)")

    typer.echo()

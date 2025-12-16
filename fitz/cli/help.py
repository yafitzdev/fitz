"""Help command for Fitz CLI."""

import typer


def command() -> None:
    """Show help and common commands."""
    typer.echo()
    typer.echo("Fitz — local-first RAG framework")
    typer.echo()
    typer.echo("Common commands:")
    typer.echo("  fitz init                    Initialize a local Fitz workspace")
    typer.echo("  fitz plugins                 List available plugins")
    typer.echo("  fitz ingest run              Ingest documents into vector DB")
    typer.echo("  fitz ingest validate         Validate documents before ingestion")
    typer.echo("  fitz pipeline query          Query your knowledge base")
    typer.echo("  fitz pipeline config show    Show resolved configuration")
    typer.echo()
    typer.echo("Quick start:")
    typer.echo("  1. fitz init")
    typer.echo("  2. fitz ingest run ./docs --collection my_docs")
    typer.echo('  3. fitz pipeline query "Your question"')
    typer.echo()
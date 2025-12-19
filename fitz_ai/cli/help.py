# fitz/cli/help.py
"""Help command for Fitz CLI."""

import typer


def command() -> None:
    """Show help and common commands."""
    typer.echo()
    typer.echo("Fitz — local-first RAG framework")
    typer.echo()
    typer.echo("Commands:")
    typer.echo("  fitz init                       Setup wizard")
    typer.echo("  fitz ingest ./docs collection   Ingest documents")
    typer.echo('  fitz query "question"           Query knowledge base')
    typer.echo("  fitz db                         Inspect collections")
    typer.echo("  fitz chunk ./doc.txt            Preview chunking")
    typer.echo("  fitz config                     Show configuration")
    typer.echo("  fitz doctor                     System diagnostics")
    typer.echo("  fitz plugins                    List all plugins")
    typer.echo()
    typer.echo("Quick start:")
    typer.echo("  1. fitz init")
    typer.echo("  2. fitz ingest ./docs my_knowledge")
    typer.echo('  3. fitz query "What is in my documents?"')
    typer.echo()
    typer.echo("More commands:")
    typer.echo("  fitz db default                 Inspect 'default' collection")
    typer.echo("  fitz chunk ./doc.txt --size 500 Preview chunking")
    typer.echo("  fitz chunk --list               List chunking plugins")
    typer.echo("  fitz ingest validate ./docs     Validate before ingesting")
    typer.echo("  fitz ingest plugins             List ingest plugins")
    typer.echo("  fitz quickstart                 Run end-to-end demo")
    typer.echo()

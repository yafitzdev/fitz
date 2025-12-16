"""Init command for Fitz CLI."""

from pathlib import Path

import typer


def command() -> None:
    """
    Initialize a local Fitz workspace.

    Creates .fitz/ if it does not exist and prints
    the active default setup.
    """
    root = Path.cwd()
    fitz_dir = root / ".fitz"

    if not fitz_dir.exists():
        fitz_dir.mkdir()
        typer.echo("✓ Created .fitz/")
    else:
        typer.echo("✓ .fitz/ already exists")

    typer.echo()
    typer.echo("Active defaults:")
    typer.echo("  Preset: local")
    typer.echo("  LLM: local (Ollama)")
    typer.echo("  Embedding: local")
    typer.echo("  Vector DB: local-faiss (disk)")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo("  fitz ingest run ./docs --collection my_docs")
    typer.echo('  fitz pipeline query "Your question"')
    typer.echo()
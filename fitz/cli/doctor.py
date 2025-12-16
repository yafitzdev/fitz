"""Doctor command for Fitz CLI."""

import subprocess
import sys
from pathlib import Path

import typer


def check_ollama_installed() -> bool:
    """Check if Ollama is installed on the system."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def command() -> None:
    """
    Run basic diagnostics for local setup.
    """
    typer.echo()
    typer.echo("Fitz doctor")
    typer.echo()

    # Python
    typer.echo(f"✓ Python {sys.version.split()[0]}")

    # .fitz
    if Path(".fitz").exists():
        typer.echo("✓ .fitz directory present")
    else:
        typer.echo("⚠ .fitz directory missing (run: fitz init)")

    # Ollama
    if check_ollama_installed():
        typer.echo("✓ Ollama detected")
    else:
        typer.echo("⚠ Ollama not detected (local LLM unavailable)")
        typer.echo("→ To enable local LLMs: https://ollama.com")

    typer.echo()
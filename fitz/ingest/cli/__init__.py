"""
CLI module for ingestion commands.

This module provides a modular command-line interface where each command
is defined in its own file for better organization and maintainability.

Available commands:
- run: Ingest documents into vector database
- list-plugins: Show available plugins
- validate: Validate documents before ingestion (dry-run)
- stats: Show statistics about a collection
"""
import typer

# Import individual commands
from fitz.ingest.cli import list_plugins, run, stats, validate

# Create main app
app = typer.Typer(
    help="Ingestion CLI commands",
    no_args_is_help=True,
)

# Register commands
app.command("run")(run.command)
app.command("list-plugins")(list_plugins.command)
app.command("validate")(validate.command)
app.command("stats")(stats.command)


if __name__ == "__main__":
    app()
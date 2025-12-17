# fitz/ingest/cli/__init__.py
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

from fitz.cli.errors import friendly_errors

# Create main app first
app = typer.Typer(
    help="Ingestion CLI commands",
    no_args_is_help=True,
)


# Import and register commands AFTER app creation to avoid circular imports
def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz.ingest.cli import list_plugins, run, stats, validate

    # Wrap commands with friendly error handling
    app.command("run")(friendly_errors(run.command))
    app.command("list-plugins")(friendly_errors(list_plugins.command))
    app.command("validate")(friendly_errors(validate.command))
    app.command("stats")(friendly_errors(stats.command))


# Register commands immediately
_register_commands()


if __name__ == "__main__":
    app()
# fitz_ai/cli/cli.py
"""
Fitz CLI - Main application.

Commands:
    fitz quickstart    Zero-friction RAG in one command (START HERE)
    fitz init          Setup wizard (for custom configuration)
    fitz ingest        Ingest documents
    fitz query         Query knowledge base
    fitz collections   Manage collections (list, info, delete)
    fitz config        View configuration
    fitz doctor        System diagnostics
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="fitz",
    help='Fitz - local-first RAG framework. Start with: fitz quickstart ./docs "your question"',
    no_args_is_help=True,
    add_completion=False,
)


def _register_commands() -> None:
    """Register all commands."""
    from fitz_ai.cli.commands import (
        collections,
        config,
        doctor,
        ingest,
        init,
        query,
        quickstart,
    )

    # Quickstart first - it's the entry point for new users
    app.command("quickstart")(quickstart.command)

    # Standard workflow commands
    app.command("init")(init.command)
    app.command("ingest")(ingest.command)
    app.command("query")(query.command)

    # Collection management (interactive)
    app.command("collections")(collections.command)

    # Utility commands
    app.command("config")(config.command)
    app.command("doctor")(doctor.command)


_register_commands()


if __name__ == "__main__":
    app()

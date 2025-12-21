# fitz_ai/cli/cli.py
"""
Fitz CLI v2 - Main application.

Clean, minimal CLI without legacy cruft.

Commands:
    fitz init       Setup wizard
    fitz ingest     Ingest documents
    fitz query      Query knowledge base
    fitz config     View configuration
    fitz doctor     System diagnostics
"""

from __future__ import annotations

import typer

app = typer.Typer(
    name="fitz",
    help="Fitz - local-first RAG framework",
    no_args_is_help=True,
    add_completion=False,
)


def _register_commands() -> None:
    """Register all commands."""
    from fitz_ai.cli.commands import config, doctor, ingest, init, query

    app.command("init")(init.command)
    app.command("ingest")(ingest.command)
    app.command("query")(query.command)
    app.command("config")(config.command)
    app.command("doctor")(doctor.command)


_register_commands()


if __name__ == "__main__":
    app()

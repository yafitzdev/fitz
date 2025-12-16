# fitz/cli/cli.py

"""
Main Fitz CLI.

Goals:
- Discoverability first
- Zero magic
- No core side effects
"""

from __future__ import annotations

import typer

app = typer.Typer(
    help="Fitz — local-first RAG framework",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# Register sub-apps with lazy imports to avoid circular dependency issues
# ---------------------------------------------------------------------------


def _register_sub_apps():
    """Register ingest and pipeline sub-apps after module initialization."""
    # Import here to avoid circular imports at module load time
    from fitz.ingest.cli import app as ingest_app
    from fitz.pipeline.cli import app as pipeline_app

    app.add_typer(ingest_app, name="ingest")
    app.add_typer(pipeline_app, name="pipeline")


# ---------------------------------------------------------------------------
# Legacy helpers (kept for compatibility)
# ---------------------------------------------------------------------------


def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz.cli import doctor, init, plugins
    from fitz.cli import help as help_module

    app.command("help")(help_module.command)
    app.command("init")(init.command)
    app.command("plugins")(plugins.command)
    app.command("doctor")(doctor.command)


# Call immediately during module initialization
_register_sub_apps()
_register_commands()


if __name__ == "__main__":
    app()
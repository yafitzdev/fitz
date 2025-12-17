# fitz/cli/cli.py

"""
Main Fitz CLI.

Goals:
- Discoverability first
- Zero magic
- User-friendly errors
"""

from __future__ import annotations

import typer

# Try to import error handler, but don't fail if not present
try:
    from fitz.cli.errors import friendly_errors, install_global_handler

    install_global_handler()
    HAS_ERROR_HANDLER = True
except ImportError:
    HAS_ERROR_HANDLER = False

    # Dummy decorator if errors.py not installed
    def friendly_errors(func):
        return func


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
    # v0.3.0: pipeline moved to engines/classic_rag/
    from fitz.engines.classic_rag.pipeline.cli import app as pipeline_app
    from fitz.ingest.cli import app as ingest_app

    app.add_typer(ingest_app, name="ingest")
    app.add_typer(pipeline_app, name="pipeline")


# ---------------------------------------------------------------------------
# Commands with friendly error handling
# ---------------------------------------------------------------------------


def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz.cli import doctor
    from fitz.cli import help as help_module
    from fitz.cli import init, plugins, quickstart

    # Wrap commands with friendly error handling
    app.command("help")(friendly_errors(help_module.command))
    app.command("init")(friendly_errors(init.command))
    app.command("plugins")(friendly_errors(plugins.command))
    app.command("doctor")(friendly_errors(doctor.command))
    app.command("quickstart")(friendly_errors(quickstart.command))


# Call immediately during module initialization
_register_sub_apps()
_register_commands()


if __name__ == "__main__":
    app()

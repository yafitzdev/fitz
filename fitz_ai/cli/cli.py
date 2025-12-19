# fitz_ai/cli/cli.py
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
    from fitz_ai.cli.errors import friendly_errors, install_global_handler

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
    # Try to import ingest CLI, but don't fail if not present
    try:
        from fitz_ai.cli import app as ingest_app
        app.add_typer(ingest_app, name="ingest")
    except (ImportError, ModuleNotFoundError):
        # Ingest CLI not available - skip it
        pass

    # Keep pipeline as hidden alias for backwards compatibility
    try:
        from fitz_ai.cli import app as pipeline_app
        app.add_typer(pipeline_app, name="pipeline", hidden=True)
    except (ImportError, ModuleNotFoundError):
        # Pipeline CLI not available - skip it
        pass


# ---------------------------------------------------------------------------
# Commands with friendly error handling
# ---------------------------------------------------------------------------


def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz_ai.cli.commands import chunk, db, doctor
    from fitz_ai.cli.commands import help as help_module
    from fitz_ai.cli.commands import init, plugins, quickstart
    from fitz_ai.cli.commands import command as config_command
    from fitz_ai.cli.commands import command as query_command

    # Wrap commands with friendly error handling
    app.command("help")(friendly_errors(help_module.command))
    app.command("init")(friendly_errors(init.command))
    app.command("plugins")(friendly_errors(plugins.command))
    app.command("doctor")(friendly_errors(doctor.command))
    app.command("quickstart")(friendly_errors(quickstart.command))

    # TOP-LEVEL query command (the main one users will use!)
    app.command("query")(friendly_errors(query_command))

    # TOP-LEVEL config command
    app.command("config")(friendly_errors(config_command))

    # Database inspection command
    app.command("db")(friendly_errors(db.command))

    # Chunking preview command
    app.command("chunk")(friendly_errors(chunk.command))


# Call immediately during module initialization
_register_sub_apps()
_register_commands()


if __name__ == "__main__":
    app()

# fitz/pipeline/cli/__init__.py
"""
CLI module for pipeline commands.

This module provides a modular command-line interface where each command
is defined in its own file for better organization and maintainability.

Available commands:
- query: Run a RAG query through the pipeline
- config show: Display resolved configuration
- test: Test pipeline setup and connections
"""

import typer

# Try to import error handler, but don't fail if not present
try:
    from fitz_ai.cli.errors import friendly_errors

    HAS_ERROR_HANDLER = True
except ImportError:
    HAS_ERROR_HANDLER = False

    def friendly_errors(func):
        return func


# Create main app first
app = typer.Typer(
    help="Pipeline CLI commands",
    no_args_is_help=True,
)

# Create config sub-app for nested commands
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")


# Import and register commands AFTER app creation to avoid circular imports
def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz_ai.engines.classic_rag.pipeline.cli import config_show, query_with_preset, test

    # Wrap commands with friendly error handling
    app.command("query")(friendly_errors(query_with_preset.command))
    config_app.command("show")(friendly_errors(config_show.command))
    app.command("test")(friendly_errors(test.command))


# Register commands immediately
_register_commands()


if __name__ == "__main__":
    app()

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

# Import individual commands
from fitz.pipeline.cli import config_show, query, test

# Create main app
app = typer.Typer(
    help="Pipeline CLI commands",
    no_args_is_help=True,
)

# Create config sub-app for nested commands
config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")

# Register commands
app.command("query")(query.command)
config_app.command("show")(config_show.command)
app.command("test")(test.command)


if __name__ == "__main__":
    app()
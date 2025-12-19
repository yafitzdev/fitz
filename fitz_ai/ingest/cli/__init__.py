# fitz_ai/ingest/cli/__init__.py
"""
CLI module for ingestion commands.

Usage:
    fitz ingest ./docs collection           # Ingest documents
    fitz ingest plugins                     # List available plugins
    fitz ingest validate ./docs             # Validate without ingesting
    fitz ingest stats collection            # Show collection stats
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

# Try to import interactive mode, but don't fail if questionary not installed
try:
    from fitz_ai.ingest.cli.interactive import run_interactive_ingestion, integrate_with_run_command

    HAS_INTERACTIVE = True
except ImportError:
    HAS_INTERACTIVE = False

# Create main app - invoke_without_command enables default command behavior
app = typer.Typer(
    help="Ingest documents into vector database.",
    invoke_without_command=True,
)


# Import and register commands AFTER app creation to avoid circular imports
def _register_commands():
    """Register commands after module initialization to avoid circular imports."""
    from fitz_ai.ingest.cli import list_plugins, run, stats, validate

    # Register the main ingest command as callback (default when no subcommand)
    @app.callback(invoke_without_command=True)
    @friendly_errors
    def main(
            ctx: typer.Context,
            source: str = typer.Argument(None, help="Source to ingest (file or directory)."),
            collection: str = typer.Argument(None, help="Target collection name."),
            ingest_plugin: str = typer.Option("local", "--ingest", "-i", help="Ingestion plugin name."),
            # Chunker options
            chunker: str = typer.Option("simple", "--chunker", "-c", help="Chunking plugin to use."),
            chunk_size: int = typer.Option(1000, "--chunk-size", help="Target chunk size in characters."),
            chunk_overlap: int = typer.Option(0, "--chunk-overlap", help="Overlap between chunks in characters."),
            min_section_chars: int = typer.Option(50, "--min-section-chars",
                                                  help="Minimum section size for section-based chunkers."),
            max_section_chars: int = typer.Option(3000, "--max-section-chars",
                                                  help="Maximum section size for section-based chunkers."),
            # Other options
            embedding_plugin: str = typer.Option(
                "cohere", "--embedding", "-e", help="Embedding plugin name."
            ),
            vector_db_plugin: str = typer.Option(
                "qdrant", "--vector-db", "-v", help="Vector DB plugin name."
            ),
            batch_size: int = typer.Option(
                50, "--batch-size", "-b", help="Batch size for vector DB writes."
            ),
            quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress progress output."),
    ):
        """
        Ingest documents into a vector database.

        Examples:
            fitz ingest ./docs default
            fitz ingest ./docs my_knowledge
            fitz ingest ./docs my_docs --embedding openai
            fitz ingest ./doc.pdf papers --chunker pdf_sections
        """
        # If a subcommand was invoked, let it handle things
        if ctx.invoked_subcommand is not None:
            return

        # If no source provided, run interactive mode (if available)
        if source is None:
            if HAS_INTERACTIVE:
                # Run interactive mode
                config = run_interactive_ingestion()
                if config:
                    integrate_with_run_command(config)
            else:
                # Fallback to help if questionary not installed
                typer.echo(ctx.get_help())
                typer.echo()
                typer.echo("💡 TIP: Install questionary for interactive mode:")
                typer.echo("   pip install questionary")

            raise typer.Exit(0)

        # Import Path here to avoid issues
        from pathlib import Path

        # Call the actual run command
        run.command(
            source=Path(source),
            collection=collection or "default",
            ingest_plugin=ingest_plugin,
            chunker=chunker,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_section_chars=min_section_chars,
            max_section_chars=max_section_chars,
            embedding_plugin=embedding_plugin,
            vector_db_plugin=vector_db_plugin,
            batch_size=batch_size,
            quiet=quiet,
        )

    # Register subcommands with cleaner names
    app.command("plugins")(friendly_errors(list_plugins.command))
    app.command("validate")(friendly_errors(validate.command))
    app.command("stats")(friendly_errors(stats.command))

    # Keep old names as hidden aliases for backwards compatibility
    app.command("run", hidden=True)(friendly_errors(run.command))
    app.command("list-plugins", hidden=True)(friendly_errors(list_plugins.command))


# Register commands immediately
_register_commands()

if __name__ == "__main__":
    app()
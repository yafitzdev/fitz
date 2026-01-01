# fitz_ai/cli/cli.py
"""
Fitz CLI - Main application.

Commands:
    fitz quickstart    Zero-friction RAG in one command (START HERE)
    fitz init          Setup wizard (for custom configuration)
    fitz ingest        Ingest documents
    fitz query         Query knowledge base
    fitz chat          Interactive conversation with your knowledge base
    fitz collections   Manage collections (list, info, delete)
    fitz map           Visualize knowledge base as interactive graph
    fitz serve         Start the REST API server
    fitz config        View configuration
    fitz doctor        System diagnostics

NOTE: Commands use lazy loading - heavy imports only happen when a command is invoked.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="fitz",
    help='Fitz - local-first RAG framework. Start with: fitz quickstart ./docs "your question"',
    no_args_is_help=True,
    add_completion=False,
)


# =============================================================================
# LAZY COMMANDS
# =============================================================================
# Each command is a thin wrapper that imports the real implementation only when invoked.
# This keeps CLI startup fast by avoiding heavy imports (torch, pydantic models, etc.).


@app.command("quickstart")
def quickstart(
    source: Optional[Path] = typer.Argument(None, help="Path to documents (file or directory)."),
    question: Optional[str] = typer.Argument(None, help="Question to ask about your documents."),
    collection: str = typer.Option("quickstart", "--collection", "-c", help="Collection name."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed progress."),
) -> None:
    """One-command RAG: ingest docs and ask a question."""
    from fitz_ai.cli.commands import quickstart as mod

    mod.command(source=source, question=question, collection=collection, verbose=verbose)


@app.command("init")
def init(
    non_interactive: bool = typer.Option(False, "--non-interactive", "-y", help="Use defaults."),
    show_config: bool = typer.Option(False, "--show", "-s", help="Preview config without saving."),
) -> None:
    """Initialize Fitz with an interactive setup wizard."""
    from fitz_ai.cli.commands import init as mod

    mod.command(non_interactive=non_interactive, show_config=show_config)


@app.command("ingest")
def ingest(
    source: Optional[str] = typer.Argument(None, help="Path to documents (file or directory)."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
    non_interactive: bool = typer.Option(False, "--yes", "-y", help="Non-interactive mode."),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-ingest all files."),
    artifacts: Optional[str] = typer.Option(None, "--artifacts", "-a", help="Artifacts to generate."),
    hierarchy: bool = typer.Option(False, "--hierarchy", "-H", help="Enable hierarchical summaries."),
) -> None:
    """Ingest documents into the knowledge base."""
    from fitz_ai.cli.commands import ingest as mod

    mod.command(source=source, collection=collection, engine=engine, non_interactive=non_interactive, force=force, artifacts=artifacts, hierarchy=hierarchy)


@app.command("query")
def query(
    question: Optional[str] = typer.Argument(None, help="Question to ask."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
) -> None:
    """Query the knowledge base."""
    from fitz_ai.cli.commands import query as mod

    mod.command(question=question, collection=collection, engine=engine)


@app.command("chat")
def chat(
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
) -> None:
    """Interactive chat with your knowledge base."""
    from fitz_ai.cli.commands import chat as mod

    mod.command(collection=collection)


@app.command("collections")
def collections() -> None:
    """Manage collections (list, info, delete)."""
    from fitz_ai.cli.commands import collections as mod

    mod.command()


@app.command("map")
def map_cmd(
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path."),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    no_open: bool = typer.Option(False, "--no-open", help="Don't open in browser."),
    rebuild: bool = typer.Option(False, "--rebuild", help="Force rebuild embeddings."),
    similarity_threshold: float = typer.Option(0.8, "--similarity-threshold", "-t", help="Minimum similarity for edges."),
    no_similarity_edges: bool = typer.Option(False, "--no-similarity-edges", help="Don't show similarity edges."),
) -> None:
    """Visualize knowledge base as interactive graph."""
    from fitz_ai.cli.commands import map as mod

    mod.command(output=output, collection=collection, no_open=no_open, rebuild=rebuild, similarity_threshold=similarity_threshold, no_similarity_edges=no_similarity_edges)


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Start the REST API server."""
    from fitz_ai.cli.commands import serve as mod

    mod.command(host=host, port=port, reload=reload)


@app.command("config")
def config(
    show_path: bool = typer.Option(False, "--path", "-p", help="Show config file path."),
    as_json: bool = typer.Option(False, "--json", help="Output as JSON."),
    raw: bool = typer.Option(False, "--raw", help="Show raw YAML."),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config in editor."),
) -> None:
    """View or edit configuration."""
    from fitz_ai.cli.commands import config as mod

    mod.command(show_path=show_path, as_json=as_json, raw=raw, edit=edit)


@app.command("doctor")
def doctor(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output."),
    test: bool = typer.Option(False, "--test", "-t", help="Run connectivity tests."),
) -> None:
    """System diagnostics and health check."""
    from fitz_ai.cli.commands import doctor as mod

    mod.command(verbose=verbose, test=test)


# =============================================================================
# ENTERPRISE PLUGIN DISCOVERY
# =============================================================================
# If fitz-ai-enterprise is installed, add its commands to the main CLI.

try:
    from fitz_ai_enterprise.cli import benchmark_app
    app.add_typer(benchmark_app, name="benchmark")
except ImportError:
    pass  # Enterprise not installed


if __name__ == "__main__":
    app()

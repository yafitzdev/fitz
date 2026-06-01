# fitz_sage/cli/cli.py
"""
Fitz CLI - Main application.

Commands:
    fitz retrieve      Retrieve governed evidence (--source to register)
    fitz answer        Generate an optional synthesized answer
    fitz query         Compatibility synthesized-answer command
    fitz collections   Manage collections (list, info, delete)
    fitz serve         Start the REST API server

Configuration: .fitz/config.yaml (auto-created on first run)

NOTE: Commands use lazy loading - heavy imports only happen when a command is invoked.
"""

from __future__ import annotations

from fitz_sage.logging import configure_logging

# Install the root log handler early (WARNING keeps CLI output clean; set
# FITZ_LOG_LEVEL via the embedding app for more). Must precede HF imports.
configure_logging()

# Platform configuration - must run before any HuggingFace imports
from fitz_sage.core.platform import configure_huggingface_windows  # noqa: E402

configure_huggingface_windows()

from pathlib import Path  # noqa: E402
from typing import Optional  # noqa: E402

import typer  # noqa: E402

app = typer.Typer(
    name="fitz",
    help='Fitz - local-first retrieval. Start with: fitz retrieve "your question" --source ./docs',
    no_args_is_help=True,
    add_completion=False,
)


# =============================================================================
# LAZY COMMANDS
# =============================================================================
# Each command is a thin wrapper that imports the real implementation only when invoked.
# This keeps CLI startup fast by avoiding heavy imports (torch, pydantic models, etc.).


@app.command("query")
def query(
    question: Optional[str] = typer.Argument(None, help="Question to ask."),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Path to documents (registers before querying)."
    ),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
    chat: bool = typer.Option(False, "--chat", help="Interactive chat mode."),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=(
            "OpenAI-compatible chat endpoint URL "
            "(e.g. http://localhost:8080/v1, https://api.openai.com/v1). "
            "Overrides chat_base_url; pairs with --model."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help=(
            "Chat model name to send to --endpoint. "
            "If --endpoint is set without --model, the engine's "
            "configured chat_smart model is used."
        ),
    ),
    api_key_env: Optional[str] = typer.Option(
        None,
        "--api-key-env",
        help=(
            "Environment variable name holding an API key for --endpoint "
            "(e.g. OPENAI_API_KEY, TOGETHER_API_KEY). Omit for "
            "unauthenticated local servers."
        ),
    ),
) -> None:
    """Query the knowledge base. Use --source to register docs, --chat for interactive mode."""
    from fitz_sage.cli.commands import query as mod

    mod.command(
        question=question,
        source=source,
        collection=collection,
        engine=engine,
        chat=chat,
        endpoint=endpoint,
        model=model,
        api_key_env=api_key_env,
    )


@app.command("retrieve")
def retrieve(
    question: Optional[str] = typer.Argument(None, help="Question to retrieve evidence for."),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Path to documents (registers before retrieval)."
    ),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
    output_format: str = typer.Option(
        "text",
        "--format",
        help="Output format: text or json.",
    ),
    top_k: Optional[int] = typer.Option(None, "--top-k", help="Maximum evidence items to show."),
) -> None:
    """Retrieve governed evidence without answer synthesis."""
    from fitz_sage.cli.commands import retrieve as mod

    mod.command(
        question=question,
        source=source,
        collection=collection,
        engine=engine,
        output_format=output_format,
        top_k=top_k,
    )


@app.command("answer")
def answer(
    question: Optional[str] = typer.Argument(None, help="Question to answer."),
    source: Optional[Path] = typer.Option(
        None, "--source", "-s", help="Path to documents (registers before answering)."
    ),
    collection: Optional[str] = typer.Option(None, "--collection", "-c", help="Collection name."),
    engine: Optional[str] = typer.Option(None, "--engine", "-e", help="Engine to use."),
    endpoint: Optional[str] = typer.Option(
        None,
        "--endpoint",
        help=(
            "OpenAI-compatible chat endpoint URL "
            "(e.g. http://localhost:8080/v1, https://api.openai.com/v1). "
            "Overrides chat_base_url; pairs with --model."
        ),
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Chat model name to send to --endpoint.",
    ),
    api_key_env: Optional[str] = typer.Option(
        None,
        "--api-key-env",
        help="Environment variable name holding an API key for --endpoint.",
    ),
) -> None:
    """Answer with optional synthesis; use retrieve for evidence-only output."""
    from fitz_sage.cli.commands import query as mod

    mod.command(
        question=question,
        source=source,
        collection=collection,
        engine=engine,
        chat=False,
        endpoint=endpoint,
        model=model,
        api_key_env=api_key_env,
    )


@app.command("collections")
def collections() -> None:
    """Manage collections (list, info, delete)."""
    from fitz_sage.cli.commands import collections as mod

    mod.command()


@app.command("serve")
def serve(
    host: str = typer.Option("127.0.0.1", "--host", "-h", help="Host to bind to."),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to."),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload."),
) -> None:
    """Start the REST API server."""
    from fitz_sage.cli.commands import serve as mod

    mod.command(host=host, port=port, reload=reload)


# =============================================================================
# SUBCOMMAND GROUPS
# =============================================================================


# =============================================================================
# ENTERPRISE PLUGIN DISCOVERY
# =============================================================================
# If fitz-sage-enterprise is installed, add its commands to the main CLI.

try:
    from fitz_sage_enterprise.cli import benchmark_app  # noqa: E402

    app.add_typer(benchmark_app, name="benchmark")
except ImportError:
    pass  # Enterprise not installed


if __name__ == "__main__":
    app()

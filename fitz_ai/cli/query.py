# fitz/cli/query.py
"""
Top-level query command.

Usage:
    fitz query "What is in my documents?"
    fitz query "Explain X" --max-sources 5
    fitz query "Question" -c custom.yaml
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import CLI, PIPELINE

logger = get_logger(__name__)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def command(
    question: str = typer.Argument(..., help="The question to answer"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use a named preset (local, openai, cohere).",
    ),
    max_sources: Optional[int] = typer.Option(
        None,
        "--max-sources",
        "-n",
        help="Maximum number of sources to retrieve.",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        "-k",
        help="Collection to query (overrides config).",
    ),
    filters: Optional[str] = typer.Option(
        None,
        "--filters",
        "-f",
        help='JSON metadata filters (e.g., \'{"topic": "physics"}\')',
    ),
) -> None:
    """
    Query your knowledge base.

    Examples:
        fitz query "What is machine learning?"
        fitz query "Explain RAG" --max-sources 5
        fitz query "Question" --collection my_docs
        fitz query "Question" --preset local
    """
    from fitz_ai.core import Constraints, GenerationError, KnowledgeError, QueryError
    from fitz_ai.engines.classic_rag.config.loader import load_config as load_rag_config
    from fitz_ai.engines.classic_rag.exceptions import LLMError
    from fitz_ai.engines.classic_rag.runtime import run_classic_rag

    # Determine config source
    config_path = None
    if preset:
        logger.info(f"{CLI}{PIPELINE} Using preset: {preset}")
        from fitz_ai.engines.classic_rag.config.presets import get_preset

        try:
            preset_dict = get_preset(preset)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        from fitz_ai.engines.classic_rag.config.schema import FitzConfig

        config_obj = FitzConfig.from_dict(preset_dict)
    else:
        config_path = str(config) if config else None
        config_source = config_path or "<default>"
        logger.info(f"{CLI}{PIPELINE} Running query with config={config_source}")
        config_obj = load_rag_config(config_path)

    # Override collection if specified
    if collection and hasattr(config_obj, "retriever"):
        config_obj.retriever.collection = collection

    # Build constraints if provided
    constraints = None
    if max_sources or filters:
        filter_dict = {}
        if filters:
            import json

            try:
                filter_dict = json.loads(filters)
            except json.JSONDecodeError as e:
                typer.echo(f"Error: Invalid JSON in --filters: {e}", err=True)
                raise typer.Exit(code=1)

        constraints = Constraints(max_sources=max_sources, filters=filter_dict)

    # Run query
    if RICH_AVAILABLE:
        console.print("[dim]Processing query...[/dim]")
    else:
        typer.echo("Processing query...")

    try:
        answer = run_classic_rag(query=question, config=config_obj, constraints=constraints)
    except LLMError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=1)
    except QueryError as e:
        typer.echo(f"Query error: {e}", err=True)
        raise typer.Exit(code=1)
    except KnowledgeError as e:
        typer.echo(f"Knowledge retrieval error: {e}", err=True)
        raise typer.Exit(code=1)
    except GenerationError as e:
        typer.echo(f"Answer generation error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        # Check if the root cause is an LLMError
        root_cause = e.__cause__
        while root_cause:
            if isinstance(root_cause, LLMError):
                typer.echo(str(root_cause), err=True)
                raise typer.Exit(code=1)
            root_cause = getattr(root_cause, "__cause__", None)

        typer.echo(f"Unexpected error: {e}", err=True)
        logger.exception("Unexpected error during query execution")
        raise typer.Exit(code=1)

    # Display answer
    typer.echo()
    if RICH_AVAILABLE:
        console.print(
            Panel(answer.text or "(No answer generated)", title="Answer", border_style="green")
        )
    else:
        typer.echo("=" * 60)
        typer.echo("ANSWER")
        typer.echo("=" * 60)
        typer.echo()
        typer.echo(answer.text or "(No answer generated)")
        typer.echo()

    # Display sources if available
    if answer.provenance:
        typer.echo()
        if RICH_AVAILABLE:
            console.print("[bold]Sources:[/bold]")
        else:
            typer.echo("SOURCES:")
            typer.echo("-" * 40)

        for i, prov in enumerate(answer.provenance, 1):
            if RICH_AVAILABLE:
                console.print(f"  [dim][{i}][/dim] {prov.source_id}")
            else:
                typer.echo(f"[{i}] {prov.source_id}")

            if prov.excerpt:
                excerpt = prov.excerpt[:150] + "..." if len(prov.excerpt) > 150 else prov.excerpt
                if RICH_AVAILABLE:
                    console.print(f"      [dim]{excerpt}[/dim]")
                else:
                    typer.echo(f"    {excerpt}")

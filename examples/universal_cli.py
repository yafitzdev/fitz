"""
Universal CLI Example - Using the universal runner API.

This demonstrates how CLI commands can now use the universal runner
to support multiple engines without code changes.
"""

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.core import Constraints, GenerationError, KnowledgeError, QueryError
from fitz_ai.runtime import list_engines, list_engines_with_info, run

app = typer.Typer(help="Universal Fitz CLI")


@app.command("query")
def query_command(
    question: str = typer.Argument(..., help="Question to ask"),
    engine: str = typer.Option(
        "classic_rag", "--engine", "-e", help="Engine to use (classic_rag, clara, etc.)"
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config file"),
    max_sources: Optional[int] = typer.Option(None, "--max-sources", help="Maximum sources to use"),
) -> None:
    """
    Execute a query using any engine.

    Examples:
        # Default engine (classic_rag)
        fitz query "What is quantum computing?"

        # Specific engine
        fitz query "Explain X" --engine clara

        # With constraints
        fitz query "What is Y?" --max-sources 5
    """
    # Build constraints
    constraints = None
    if max_sources:
        constraints = Constraints(max_sources=max_sources)

    # Run query
    typer.echo(f"Using engine: {engine}")
    typer.echo("Processing query...\n")

    try:
        answer = run(query=question, engine=engine, config_path=config, constraints=constraints)
    except QueryError as e:
        typer.echo(f"❌ Query error: {e}", err=True)
        raise typer.Exit(code=1)
    except KnowledgeError as e:
        typer.echo(f"❌ Knowledge error: {e}", err=True)
        raise typer.Exit(code=1)
    except GenerationError as e:
        typer.echo(f"❌ Generation error: {e}", err=True)
        raise typer.Exit(code=1)
    except Exception as e:
        typer.echo(f"❌ Error: {e}", err=True)
        raise typer.Exit(code=1)

    # Display answer
    typer.echo("=" * 60)
    typer.echo("ANSWER")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo(answer.text)
    typer.echo()

    # Display sources
    if answer.provenance:
        typer.echo("=" * 60)
        typer.echo("SOURCES")
        typer.echo("=" * 60)
        typer.echo()
        for i, prov in enumerate(answer.provenance, 1):
            typer.echo(f"[{i}] {prov.source_id}")
            if prov.excerpt:
                excerpt = prov.excerpt[:150] + "..." if len(prov.excerpt) > 150 else prov.excerpt
                typer.echo(f"    {excerpt}")
            typer.echo()


@app.command("engines")
def list_engines_command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show descriptions")
) -> None:
    """
    List available engines.

    Examples:
        # Simple list
        fitz engines

        # With descriptions
        fitz engines --verbose
    """
    if verbose:
        typer.echo("Available engines:\n")
        for name, desc in list_engines_with_info().items():
            typer.echo(f"  {name}")
            typer.echo(f"    {desc}\n")
    else:
        engines = list_engines()
        typer.echo("Available engines:")
        for name in engines:
            typer.echo(f"  - {name}")


if __name__ == "__main__":
    app()

# fitz/engines/classic_rag/pipeline/cli/query_with_preset.py
"""
Query command with preset support.

This module provides the main query command for the RAG pipeline,
with support for configuration presets and constraints.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz.core import (
    Constraints,
    GenerationError,
    KnowledgeError,
    QueryError,
)
from fitz.engines.classic_rag.config.loader import load_config as load_rag_config
from fitz.engines.classic_rag.errors.llm import LLMError
from fitz.engines.classic_rag.runtime import run_classic_rag
from fitz.logging.logger import get_logger
from fitz.logging.tags import CLI, PIPELINE

logger = get_logger(__name__)


def command(
    question: str = typer.Argument(..., help="The question to answer"),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML file. Overrides preset if both specified.",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use a named preset (e.g., 'local' for offline mode). "
        "Available: local, openai, azure, cohere",
    ),
    max_sources: Optional[int] = typer.Option(
        None,
        "--max-sources",
        "-n",
        help="Maximum number of sources to retrieve.",
    ),
    filters: Optional[str] = typer.Option(
        None,
        "--filters",
        "-f",
        help='JSON string of metadata filters (e.g., \'{"topic": "physics"}\')',
    ),
) -> None:
    """
    Run a RAG query through the pipeline.

    This command:
    1. Loads configuration (from file, preset, or defaults)
    2. Retrieves relevant chunks from vector database
    3. Builds context and prompt
    4. Calls LLM for answer generation
    5. Returns answer with citations (if enabled)

    Examples:
        # Basic query with default config
        fitz-pipeline query "What are the main topics?"

        # Query with custom config
        fitz-pipeline query "Explain concept X" --config custom.yaml

        # Query with preset (offline/no API keys needed)
        fitz-pipeline query "What is this?" --preset local

        # Query with constraints
        fitz-pipeline query "What is X?" --max-sources 5

        # Query with metadata filters
        fitz-pipeline query "Explain Y" --filters '{"topic": "physics"}'
    """
    # Determine config source
    config_path = None
    if preset:
        logger.info(f"{CLI}{PIPELINE} Using preset: {preset}")
        # Presets are now handled by config loader
        from fitz.engines.classic_rag.config.presets import get_preset

        try:
            preset_dict = get_preset(preset)
        except ValueError as e:
            typer.echo(f"Error: {e}", err=True)
            raise typer.Exit(code=1)

        # Load config from preset dict
        from fitz.engines.classic_rag.config.schema import FitzConfig

        config_obj = FitzConfig.from_dict(preset_dict)
    else:
        config_path = str(config) if config else None
        config_source = config_path or "<default>"
        logger.info(f"{CLI}{PIPELINE} Running RAG query with config={config_source}")
        config_obj = load_rag_config(config_path)

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

    # Run query using new runtime
    typer.echo("Processing query...")
    try:
        answer = run_classic_rag(query=question, config=config_obj, constraints=constraints)
    except LLMError as e:
        # LLMError contains user-friendly setup instructions - display them directly
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
    typer.echo("=" * 60)
    typer.echo("ANSWER")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo(answer.text or "(No answer generated)")
    typer.echo()

    # Display sources if available
    if answer.provenance:
        typer.echo("=" * 60)
        typer.echo("SOURCES")
        typer.echo("=" * 60)
        typer.echo()
        for i, prov in enumerate(answer.provenance, 1):
            typer.echo(f"[{i}] {prov.source_id}")
            if prov.excerpt:
                # Truncate long excerpts
                excerpt = prov.excerpt[:200] + "..." if len(prov.excerpt) > 200 else prov.excerpt
                typer.echo(f"    {excerpt}")
            if prov.metadata:
                # Show relevant metadata
                for key in ["title", "source", "path", "relevance_score"]:
                    if key in prov.metadata:
                        typer.echo(f"    {key}: {prov.metadata[key]}")
            typer.echo()

    # Show metadata if verbose logging
    if logger.level <= 10:  # DEBUG level
        typer.echo("=" * 60)
        typer.echo("METADATA")
        typer.echo("=" * 60)
        typer.echo()
        import json

        typer.echo(json.dumps(answer.metadata, indent=2))

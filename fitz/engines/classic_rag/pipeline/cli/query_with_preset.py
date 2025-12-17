"""
Query command: Run a RAG query through the pipeline.

Usage:
    fitz-pipeline query "What is this about?"
    fitz-pipeline query "Explain X" --config my_config.yaml
    fitz-pipeline query "Explain X" --preset local
"""

from pathlib import Path
from typing import Optional

import typer

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CLI, PIPELINE
from fitz.pipeline.pipeline.engine import create_pipeline_from_yaml

logger = get_logger(__name__)


def command(
    question: str = typer.Argument(
        ...,
        help="User question to run through the RAG pipeline.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to pipeline YAML config.",
    ),
    preset: Optional[str] = typer.Option(
        None,
        "--preset",
        "-p",
        help="Use a configuration preset (local, dev, production). Overrides --config.",
    ),
) -> None:
    """
    Run a RAG query through the pipeline.

    This command:
    1. Loads the pipeline configuration (from preset or file)
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

        # Query with dev preset (cost-effective APIs)
        fitz-pipeline query "Analyze this" --preset dev

        # Query with production preset (best models)
        fitz-pipeline query "Important query" --preset production
    """
    # Handle preset vs config
    config_source = None
    if preset:
        config_source = f"preset={preset}"
        logger.info(f"{CLI}{PIPELINE} Using preset: {preset}")

        # Import here to avoid circular dependency
        from fitz.engines.classic_rag.config.presets import get_preset

        try:
            preset_config = get_preset(preset)
        except ValueError as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(code=1)

        # For now, we need to save preset to a temp file
        # TODO: Enhance create_pipeline_from_yaml to accept dict directly
        import tempfile

        import yaml

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(preset_config, f)
            temp_config_path = f.name

        try:
            pipeline = create_pipeline_from_yaml(temp_config_path)
        finally:
            import os

            os.unlink(temp_config_path)
    else:
        config_source = config if config is not None else "<default>"
        logger.info(f"{CLI}{PIPELINE} Running RAG query with config={config_source}")
        pipeline = create_pipeline_from_yaml(str(config) if config is not None else None)

    # Run query
    typer.echo("Processing query...")
    rgs_answer = pipeline.run(question)

    # Display answer
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("ANSWER")
    typer.echo("=" * 60)
    typer.echo()
    typer.echo(getattr(rgs_answer, "answer", "") or "(No answer generated)")
    typer.echo()

    # Display citations if available
    citations = getattr(rgs_answer, "citations", None)
    if citations:
        typer.echo("=" * 60)
        typer.echo("SOURCES")
        typer.echo("=" * 60)
        typer.echo()
        for c in citations:
            label = getattr(c, "label", None) or getattr(c, "source_id", "")
            title = getattr(c, "title", "") or ""
            src = ""
            meta = getattr(c, "metadata", None) or {}
            if isinstance(meta, dict):
                src = meta.get("source") or meta.get("path") or ""

            typer.echo(f"[{label}] {title}")
            if src:
                typer.echo(f"     Source: {src}")
            typer.echo()

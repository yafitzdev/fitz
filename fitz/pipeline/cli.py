# pipeline/cli.py
"""
Command-line interface for fitz_rag.

Commands:

- config show   → print effective RAG config
- query         → run a one-off RAG query
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CLI, PIPELINE
from fitz.pipeline.config.loader import load_config
from fitz.pipeline.pipeline.engine import create_pipeline_from_yaml

logger = get_logger(__name__)

app = typer.Typer(help="fitz-pipeline CLI")

config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to RAG YAML config. If omitted, built-in default.yaml is used.",
    )
) -> None:
    """
    Print the resolved RAG configuration.
    """
    logger.info(
        f"{CLI}{PIPELINE} Showing RAG config from "
        f"{config if config is not None else '<built-in default>'}"
    )

    raw_cfg = load_config(str(config) if config is not None else None)
    # Don't rely on schema helper methods (from_dict/json) that may not exist.
    typer.echo(raw_cfg)


@app.command("query")
def query(
    question: str = typer.Argument(
        ...,
        help="User question to run through the RAG pipeline.",
    ),
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to RAG YAML config used to construct the pipeline.",
    ),
) -> None:
    """
    Run a one-off RAG query.
    """
    logger.info(
        f"{CLI}{PIPELINE} Running RAG query with "
        f"config={config if config is not None else '<built-in default>'}"
    )

    pipeline = create_pipeline_from_yaml(str(config) if config is not None else None)
    rgs_answer = pipeline.run(question)

    typer.echo("# Answer\n")
    typer.echo(getattr(rgs_answer, "answer", "") or "")

    citations = getattr(rgs_answer, "citations", None)
    if citations:
        typer.echo("\n# Sources\n")
        for c in citations:
            label = getattr(c, "label", None) or getattr(c, "source_id", "")
            title = getattr(c, "title", "") or ""
            src = ""
            meta = getattr(c, "metadata", None) or {}
            if isinstance(meta, dict):
                src = meta.get("source") or meta.get("path") or ""
            typer.echo(f"- {label}: {title} {f'[{src}]' if src else ''}")


if __name__ == "__main__":
    app()

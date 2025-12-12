# src/fitz_rag/cli.py
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

from core.logging.logger import get_logger
from core.logging.tags import CLI, PIPELINE

from rag.config.loader import load_config
from rag.config.schema import RAGConfig
from rag.pipeline.engine import create_pipeline_from_yaml

logger = get_logger(__name__)

app = typer.Typer(help="fitz-rag CLI")

config_app = typer.Typer(help="Configuration commands")
app.add_typer(config_app, name="config")


@config_app.command("show")
def config_show(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to RAG YAML config. If omitted, default search locations are used.",
    )
) -> None:
    """
    Print the resolved RAG configuration.
    """
    logger.info(
        f"{CLI}{PIPELINE} Showing RAG config from "
        f"{config if config is not None else '<default search>'}"
    )

    raw_cfg = load_config(str(config) if config is not None else None)
    cfg = RAGConfig.from_dict(raw_cfg)

    typer.echo(cfg.json(indent=2))


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
        f"config={config if config is not None else '<default search>'}"
    )

    pipeline = create_pipeline_from_yaml(str(config) if config is not None else None)
    rgs_answer = pipeline.run(question)

    typer.echo("# Answer\n")
    typer.echo(rgs_answer.answer or "")

    if getattr(rgs_answer, "citations", None):
        typer.echo("\n# Sources\n")
        for c in rgs_answer.citations:
            label = getattr(c, "label", None) or getattr(c, "source_id", "")
            title = getattr(c, "title", "") or ""
            src = ""
            meta = getattr(c, "metadata", None) or {}
            if isinstance(meta, dict):
                src = meta.get("source") or meta.get("path") or ""
            typer.echo(f"- {label}: {title} {f'[{src}]' if src else ''}")


if __name__ == "__main__":
    app()

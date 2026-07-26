"""Inspect and replay versioned retrieval execution records."""

from __future__ import annotations

from pathlib import Path

import typer

from fitz_sage.cli.ui import ui
from fitz_sage.core import RetrievalRun
from fitz_sage.logging.logger import get_logger
from fitz_sage.runtime import replay_governance

logger = get_logger(__name__)


def explain_command(trace_file: Path) -> None:
    """Print a deterministic explanation of a retrieval trace."""
    try:
        print(RetrievalRun.read(trace_file).explain())
    except Exception as exc:
        ui.error(f"Explain failed: {exc}")
        logger.debug("Retrieval trace explanation failed", exc_info=True)
        raise typer.Exit(1) from exc


def replay_command(
    trace_file: Path,
    *,
    governance: str | None = None,
    output: Path | None = None,
    output_format: str = "text",
    include_content: bool = False,
) -> None:
    """Replay governance over the trace's verified frozen evidence."""
    output_format = output_format.lower().strip()
    if output_format not in {"text", "json"}:
        ui.error("--format must be 'text' or 'json'.")
        raise typer.Exit(1)

    try:
        result = replay_governance(trace_file, governance)
        if output is not None:
            result.write(output, include_content=include_content)
        if output_format == "json":
            print(result.to_json(include_content=include_content))
        else:
            print(result.explain())
    except Exception as exc:
        ui.error(f"Replay failed: {exc}")
        logger.debug("Governance replay failed", exc_info=True)
        raise typer.Exit(1) from exc


__all__ = ["explain_command", "replay_command"]

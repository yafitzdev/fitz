# fitz_sage/cli/commands/index_daemon.py
"""Hidden CLI command for detached background indexing."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_sage.cli.ui import ui
from fitz_sage.runtime import create_engine, get_default_engine, get_engine_registry


def _pid_path(collection: str) -> Path:
    """Return the PID file for this collection in the current workspace."""
    return Path.cwd() / ".fitz" / "collections" / collection / "index_daemon.pid"


def command(collection: str, engine: Optional[str]) -> None:
    """Continue indexing a persisted collection until deep enrichment is complete."""
    registry = get_engine_registry()
    engine_name = engine or get_default_engine()
    if engine_name not in registry.list():
        ui.error(f"Unknown engine: '{engine_name}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    engine_instance = create_engine(engine_name)
    engine_instance.load(collection)
    try:
        engine_instance.continue_indexing()
    finally:
        try:
            _pid_path(collection).unlink(missing_ok=True)
        except OSError:
            pass

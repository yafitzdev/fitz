# fitz_sage/cli/commands/enrichment_daemon.py
"""Hidden CLI command for detached background enrichment."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from fitz_sage.cli.ui import ui
from fitz_sage.runtime import create_engine, get_default_engine, get_engine_registry

logger = logging.getLogger(__name__)


def _pid_path(collection: str) -> Path:
    """Return the PID file for this collection in the current workspace."""
    return Path.cwd() / ".fitz" / "collections" / collection / "enrichment_daemon.pid"


def command(collection: str, engine: Optional[str]) -> None:
    """Continue enriching a persisted collection until background work settles."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
        force=True,
    )
    registry = get_engine_registry()
    engine_name = engine or get_default_engine()
    if engine_name not in registry.list():
        ui.error(f"Unknown engine: '{engine_name}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    engine_instance = create_engine(engine_name)
    try:
        logger.info("Enrichment daemon started: collection=%s engine=%s", collection, engine_name)
        logger.info("Enrichment daemon loading collection: %s", collection)
        engine_instance.load(collection)
        logger.info("Enrichment daemon continuing: %s", collection)
        engine_instance.continue_enrichment()
        logger.info("Enrichment daemon completed: collection=%s engine=%s", collection, engine_name)
    finally:
        try:
            _pid_path(collection).unlink(missing_ok=True)
        except OSError:
            pass

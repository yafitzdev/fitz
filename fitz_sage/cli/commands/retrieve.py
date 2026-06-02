# fitz_sage/cli/commands/retrieve.py
"""Retrieve governed evidence without answer synthesis."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_sage.cli.ui import display_evidence_pack, ui
from fitz_sage.core import Query
from fitz_sage.logging.logger import get_logger
from fitz_sage.runtime import create_engine, get_default_engine, get_engine_registry

logger = get_logger(__name__)


def command(
    question: Optional[str],
    source: Optional[Path],
    collection: Optional[str],
    engine: Optional[str],
    *,
    output_format: str = "text",
    top_k: int | None = None,
) -> None:
    """Run retrieval-first evidence mode."""
    output_format = output_format.lower().strip()
    if output_format not in {"text", "json"}:
        ui.error("--format must be 'text' or 'json'.")
        raise typer.Exit(1)
    if top_k is not None and top_k < 1:
        ui.error("--top-k must be greater than zero.")
        raise typer.Exit(1)
    if source is not None and not source.exists():
        ui.error(f"Path does not exist: {source}")
        raise typer.Exit(1)

    registry = get_engine_registry()
    engine_name = engine or get_default_engine()
    if engine_name not in registry.list():
        ui.error(f"Unknown engine: '{engine_name}'. Available: {', '.join(registry.list())}")
        raise typer.Exit(1)

    caps = registry.get_capabilities(engine_name)
    if not caps.supports_persistent_ingest:
        ui.error(f"Engine '{engine_name}' does not support retrieval evidence mode.")
        raise typer.Exit(1)

    selected_collection = _select_collection(registry, engine_name, collection, source)
    question_text = question if question is not None else ui.prompt_text("Question")
    metadata = {"top_k": top_k} if top_k is not None else {}
    progress = None if output_format == "json" else ui.info

    try:
        engine_instance = create_engine(engine_name)
        if source is not None:
            if progress:
                progress(f"Registering {source}...")
            engine_instance.point(source, selected_collection, progress=progress)
            engine_instance.wait_for_query_surface(progress=progress)
        else:
            if progress:
                progress(f"Loading collection '{selected_collection}'...")
            engine_instance.load(selected_collection)

        pack = engine_instance.evidence(
            Query(text=question_text, metadata=metadata),
            progress=progress,
            top_k=top_k,
        )
        if output_format == "json":
            print(pack.to_json())
        else:
            display_evidence_pack(pack, max_items=top_k or 10)
    except Exception as e:
        ui.error(f"Retrieve failed: {e}")
        logger.debug("Retrieve error", exc_info=True)
        raise typer.Exit(1)


def _select_collection(
    registry,
    engine_name: str,
    requested: Optional[str],
    source: Optional[Path],
) -> str:
    """Resolve target collection for source or collection retrieval."""
    if source is not None:
        return requested or "default"

    collections = registry.get_list_collections(engine_name)
    if not collections:
        ui.warning("No collections found.")
        ui.info("Run 'fitz retrieve \"question\" --source ./docs' to get started.")
        raise typer.Exit(0)

    if requested is not None:
        if requested not in collections:
            ui.error(f"Collection '{requested}' not found. Available: {', '.join(collections)}")
            raise typer.Exit(1)
        return requested

    if len(collections) == 1:
        return collections[0]

    return ui.prompt_numbered_choice("Collection", collections, collections[0])

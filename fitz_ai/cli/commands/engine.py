# fitz_ai/cli/commands/engine.py
"""
Engine management command.

Allows users to view and set the default engine for all CLI commands.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.cli.ui import ui
from fitz_ai.core.paths import FitzPaths
from fitz_ai.runtime import get_default_engine, get_engine_registry, list_engines


def command(
    name: Optional[str] = None,
    list_available: bool = False,
) -> None:
    """
    View or set the default engine.

    Args:
        name: Engine name to set as default (optional)
        list_available: List all available engines
    """
    registry = get_engine_registry()
    available = list_engines()
    descriptions = registry.list_with_descriptions()
    current = get_default_engine()

    # List mode
    if list_available:
        _list_engines(available, descriptions, current)
        return

    # No argument - interactive selection
    if name is None:
        _interactive_selection(available, descriptions, current)
        return

    # Direct set mode
    if name not in available:
        ui.error(f"Unknown engine: '{name}'. Available: {', '.join(available)}")
        raise typer.Exit(1)

    _set_default_engine(name, current)


def _list_engines(
    available: list[str],
    descriptions: dict[str, str],
    current: str,
) -> None:
    """List all available engines."""
    ui.header("Available Engines")
    print()
    for engine in available:
        desc = descriptions.get(engine, "")
        marker = " (current)" if engine == current else ""
        if marker:
            ui.info(f"  {engine}{marker}")
        else:
            print(f"  {engine}")
        if desc:
            print(f"    {desc}")
    print()


def _interactive_selection(
    available: list[str],
    descriptions: dict[str, str],
    current: str,
) -> None:
    """Interactive engine selection using the card UI."""
    ui.header("Engine Selection")
    ui.info(f"Current default: {current}")
    print()

    selected = ui.prompt_engine_selection(
        engines=available,
        descriptions=descriptions,
        default=current,
    )

    _set_default_engine(selected, current)


def _set_default_engine(engine: str, current: str) -> None:
    """Set the default engine in config.yaml."""
    if engine == current:
        ui.info(f"Engine already set to: {engine}")
        return

    # Load or create config
    config_path = FitzPaths.config()
    config = _load_config(config_path)

    # Update default_engine
    config["default_engine"] = engine

    # Save
    _save_config(config_path, config)

    ui.success(f"Default engine set to: {engine}")
    ui.info("All CLI commands will now use this engine by default.")


def _load_config(path: Path) -> dict:
    """Load config from YAML file."""
    if not path.exists():
        return {}

    try:
        import yaml

        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _save_config(path: Path, config: dict) -> None:
    """Save config to YAML file."""
    import yaml

    # Ensure parent directory exists
    FitzPaths.ensure_workspace()

    with open(path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

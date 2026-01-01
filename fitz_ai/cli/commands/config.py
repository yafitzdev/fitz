# fitz_ai/cli/commands/config.py
"""
Configuration command.

Usage:
    fitz config                # Show current config
    fitz config --json         # Output as JSON
    fitz config --path         # Show config file path
    fitz config --edit         # Open config in editor
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, Table, console, ui
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Display Functions
# =============================================================================


def _show_config_summary(ctx: CLIContext) -> None:
    """Show a summary table of config settings."""
    print()

    if RICH:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Plugin", style="green")
        table.add_column("Details", style="dim")

        # Chat
        table.add_row("Chat", ctx.chat_plugin or "?", ctx.chat_model_smart)

        # Embedding
        table.add_row("Embedding", ctx.embedding_plugin or "?", ctx.embedding_model)

        # Vector DB
        vdb_host = ctx.vector_db_kwargs.get("host", "")
        vdb_port = ctx.vector_db_kwargs.get("port", "")
        vdb_details = f"{vdb_host}:{vdb_port}" if vdb_host else ""
        table.add_row("Vector DB", ctx.vector_db_plugin or "?", vdb_details)

        # Retriever
        table.add_row(
            "Retriever",
            ctx.retrieval_plugin,
            f"collection={ctx.retrieval_collection}, top_k={ctx.retrieval_top_k}",
        )

        # Rerank
        if ctx.rerank_enabled:
            table.add_row("Rerank", ctx.rerank_plugin or "?", ctx.rerank_model)
        else:
            table.add_row("Rerank", "[dim]disabled[/dim]", "")

        # RGS
        citations = "on" if ctx.rgs_citations else "off"
        grounding = "strict" if ctx.rgs_strict_grounding else "relaxed"
        table.add_row("RGS", f"citations={citations}", f"grounding={grounding}")

        console.print(table)
    else:
        print("Configuration Summary:")
        print("-" * 40)

        chat_detail = f" ({ctx.chat_model_smart})" if ctx.chat_model_smart else ""
        print(f"  Chat:      {ctx.chat_plugin or '?'}{chat_detail}")

        print(f"  Embedding: {ctx.embedding_plugin or '?'}")

        print(f"  Vector DB: {ctx.vector_db_plugin or '?'}")

        print(f"  Retriever: {ctx.retrieval_plugin} (collection={ctx.retrieval_collection})")

        if ctx.rerank_enabled:
            print(f"  Rerank:    {ctx.rerank_plugin or '?'}")
        else:
            print("  Rerank:    disabled")

    print()


def _show_config_yaml(config_path: Path) -> None:
    """Show raw YAML config."""
    try:
        content = config_path.read_text()
    except Exception as e:
        ui.error(f"Failed to read config: {e}")
        return

    print()
    ui.syntax(content, "yaml")
    print()


def _show_config_json(config: dict) -> None:
    """Show config as JSON."""
    print()
    if RICH:
        json_str = json.dumps(config, indent=2)
        ui.syntax(json_str, "json")
    else:
        print(json.dumps(config, indent=2))
    print()


def _open_in_editor(config_path: Path) -> None:
    """Open config file in default editor."""
    # Try common editors
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL")

    if not editor:
        # Try to find a common editor
        for candidate in ["code", "nano", "vim", "vi", "notepad"]:
            try:
                result = subprocess.run(
                    ["which", candidate] if os.name != "nt" else ["where", candidate],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    editor = candidate
                    break
            except Exception:
                continue

    if not editor:
        ui.error("No editor found. Set EDITOR environment variable.")
        ui.info(f"Config path: {config_path}")
        return

    try:
        ui.info(f"Opening in {editor}...")
        subprocess.run([editor, str(config_path)])
    except Exception as e:
        ui.error(f"Failed to open editor: {e}")
        ui.info(f"Config path: {config_path}")


# =============================================================================
# Main Command
# =============================================================================


def _get_config_path() -> Path:
    """Get config path, checking engine-specific first, then global."""
    ctx = CLIContext.load_or_none()
    if ctx is not None:
        return ctx.config_path

    # Fall back to expected path for error messages
    engine_config = FitzPaths.engine_config("fitz_rag")
    if engine_config.exists():
        return engine_config
    return FitzPaths.config()


def command(
    show_path: bool = typer.Option(
        False,
        "--path",
        "-p",
        help="Show config file path only.",
    ),
    as_json: bool = typer.Option(
        False,
        "--json",
        "-j",
        help="Output as JSON.",
    ),
    raw: bool = typer.Option(
        False,
        "--raw",
        "-r",
        help="Show raw YAML file.",
    ),
    edit: bool = typer.Option(
        False,
        "--edit",
        "-e",
        help="Open config in editor.",
    ),
) -> None:
    """
    View and manage Fitz configuration.

    Shows a summary of your current configuration settings.

    Examples:
        fitz config              # Show config summary
        fitz config --raw        # Show raw YAML
        fitz config --json       # Output as JSON
        fitz config --path       # Show file path
        fitz config --edit       # Open in editor
    """
    # =========================================================================
    # Get config path
    # =========================================================================

    config_path = _get_config_path()

    # =========================================================================
    # Path only mode
    # =========================================================================

    if show_path:
        if config_path.exists():
            print(str(config_path))
        else:
            ui.error("Config not found.")
            ui.info("Run 'fitz quickstart' or 'fitz init' to create one.")
            raise typer.Exit(1)
        return

    # =========================================================================
    # Edit mode
    # =========================================================================

    if edit:
        if not config_path.exists():
            ui.error("No config file to edit.")
            ui.info("Run 'fitz quickstart' or 'fitz init' first.")
            raise typer.Exit(1)

        _open_in_editor(config_path)
        return

    # =========================================================================
    # Load config via CLIContext
    # =========================================================================

    ctx = CLIContext.load_or_none()
    if ctx is None:
        ui.error("No configuration found.")
        print()
        ui.info(f"Looked for: {FitzPaths.engine_config('fitz_rag')}")
        ui.info(f"        or: {FitzPaths.config()}")
        ui.info("Run 'fitz quickstart' or 'fitz init' to create a configuration.")
        raise typer.Exit(1)

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Config", "Check your RAG configuration")
    ui.info(f"File: {ctx.config_path}")

    # =========================================================================
    # JSON output
    # =========================================================================

    if as_json:
        _show_config_json(ctx.raw_config)
        return

    # =========================================================================
    # Raw YAML output
    # =========================================================================

    if raw:
        _show_config_yaml(ctx.config_path)
        return

    # =========================================================================
    # Summary view (default)
    # =========================================================================

    _show_config_summary(ctx)

    # Show helpful hints
    ui.info("Tip: Use --raw for full YAML, --edit to modify")

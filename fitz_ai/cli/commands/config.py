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

from fitz_ai.cli.ui import RICH, Table, console, ui
from fitz_ai.core.config import load_config_dict
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Display Functions
# =============================================================================


def _show_config_summary(config: dict) -> None:
    """Show a summary table of config settings."""
    print()

    if RICH:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Component", style="cyan")
        table.add_column("Plugin", style="green")
        table.add_column("Details", style="dim")

        # Chat
        chat = config.get("chat", {})
        chat_plugin = chat.get("plugin_name", "?")
        chat_model = chat.get("kwargs", {}).get("model", "")
        table.add_row("Chat", chat_plugin, chat_model)

        # Embedding
        emb = config.get("embedding", {})
        emb_plugin = emb.get("plugin_name", "?")
        emb_model = emb.get("kwargs", {}).get("model", "")
        table.add_row("Embedding", emb_plugin, emb_model)

        # Vector DB
        vdb = config.get("vector_db", {})
        vdb_plugin = vdb.get("plugin_name", "?")
        vdb_host = vdb.get("kwargs", {}).get("host", "")
        vdb_port = vdb.get("kwargs", {}).get("port", "")
        vdb_details = f"{vdb_host}:{vdb_port}" if vdb_host else ""
        table.add_row("Vector DB", vdb_plugin, vdb_details)

        # Retriever
        ret = config.get("retrieval", {})
        ret_plugin = ret.get("plugin_name", "dense")
        ret_collection = ret.get("collection", "default")
        ret_top_k = ret.get("top_k", 5)
        table.add_row(
            "Retriever", ret_plugin, f"collection={ret_collection}, top_k={ret_top_k}"
        )

        # Rerank
        rerank = config.get("rerank", {})
        rerank_enabled = rerank.get("enabled", False)
        if rerank_enabled:
            rerank_plugin = rerank.get("plugin_name", "?")
            rerank_model = rerank.get("kwargs", {}).get("model", "")
            table.add_row("Rerank", rerank_plugin, rerank_model)
        else:
            table.add_row("Rerank", "[dim]disabled[/dim]", "")

        # RGS
        rgs = config.get("rgs", {})
        citations = "on" if rgs.get("enable_citations", True) else "off"
        grounding = "strict" if rgs.get("strict_grounding", True) else "relaxed"
        table.add_row("RGS", f"citations={citations}", f"grounding={grounding}")

        console.print(table)
    else:
        print("Configuration Summary:")
        print("-" * 40)

        chat = config.get("chat", {})
        print(f"  Chat:      {chat.get('plugin_name', '?')}")

        emb = config.get("embedding", {})
        print(f"  Embedding: {emb.get('plugin_name', '?')}")

        vdb = config.get("vector_db", {})
        print(f"  Vector DB: {vdb.get('plugin_name', '?')}")

        ret = config.get("retrieval", {})
        print(
            f"  Retriever: {ret.get('plugin_name', 'dense')} (collection={ret.get('collection', 'default')})"
        )

        rerank = config.get("rerank", {})
        if rerank.get("enabled"):
            print(f"  Rerank:    {rerank.get('plugin_name', '?')}")
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

    config_path = FitzPaths.config()

    # =========================================================================
    # Path only mode
    # =========================================================================

    if show_path:
        if config_path.exists():
            print(str(config_path))
        else:
            ui.error(f"Config not found: {config_path}")
            ui.info("Run 'fitz init' to create one.")
            raise typer.Exit(1)
        return

    # =========================================================================
    # Edit mode
    # =========================================================================

    if edit:
        if not config_path.exists():
            ui.error("No config file to edit.")
            ui.info("Run 'fitz init' first.")
            raise typer.Exit(1)

        _open_in_editor(config_path)
        return

    # =========================================================================
    # Load config
    # =========================================================================

    if not config_path.exists():
        ui.error("No configuration found.")
        print()
        ui.info(f"Expected: {config_path}")
        ui.info("Run 'fitz init' to create a configuration.")
        raise typer.Exit(1)

    try:
        config = load_config_dict(config_path)
    except Exception as e:
        ui.error(f"Failed to load config: {e}")
        raise typer.Exit(1)

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Configuration")
    ui.info(f"File: {config_path}")

    # =========================================================================
    # JSON output
    # =========================================================================

    if as_json:
        _show_config_json(config)
        return

    # =========================================================================
    # Raw YAML output
    # =========================================================================

    if raw:
        _show_config_yaml(config_path)
        return

    # =========================================================================
    # Summary view (default)
    # =========================================================================

    _show_config_summary(config)

    # Show helpful hints
    ui.info("Tip: Use --raw for full YAML, --edit to modify")

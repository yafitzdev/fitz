# fitz_sage/cli/commands/config.py
"""
Configuration command — view config, run diagnostics, test connections.

Usage:
    fitz config                # Show current config
    fitz config --doctor       # Run system diagnostics
    fitz config --test         # Test actual connections
    fitz config --json         # Output as JSON
    fitz config --path         # Show config file path
    fitz config --edit         # Open config in editor
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import typer

from fitz_sage.cli.context import CLIContext
from fitz_sage.cli.ui import RICH, Panel, Table, console, ui
from fitz_sage.core.paths import FitzPaths
from fitz_sage.logging.logger import get_logger

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
        vdb_host = getattr(ctx.vector_db_kwargs, "host", None) or ""
        vdb_port = getattr(ctx.vector_db_kwargs, "port", None) or ""
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
            except Exception as e:
                logger.debug(f"Editor {candidate} not found: {e}")
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
    """Get config path."""
    ctx = CLIContext.load()
    if ctx.config_path is not None:
        return ctx.config_path

    return FitzPaths.config()


def command(
    show_path: bool = typer.Option(False, "--path", "-p", help="Show config file path only."),
    as_json: bool = typer.Option(False, "--json", "-j", help="Output as JSON."),
    raw: bool = typer.Option(False, "--raw", "-r", help="Show raw YAML file."),
    edit: bool = typer.Option(False, "--edit", "-e", help="Open config in editor."),
    doctor: bool = typer.Option(False, "--doctor", "-d", help="Run system diagnostics."),
    test: bool = typer.Option(False, "--test", "-t", help="Test actual connections."),
) -> None:
    """
    View configuration and run diagnostics.

    Examples:
        fitz config              # Show config summary
        fitz config --doctor     # System diagnostics
        fitz config --test       # Test connections
        fitz config --edit       # Open in editor
    """
    # =========================================================================
    # Doctor mode
    # =========================================================================

    if doctor or test:
        _run_doctor(test=test)
        return

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
    # Load config via CLIContext (always succeeds with defaults)
    # =========================================================================

    ctx = CLIContext.load()

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

    ui.info(f"Source: {ctx.config_source}")
    _show_config_summary(ctx)
    ui.info("Tip: Use --raw for full YAML, --edit to modify, --doctor for diagnostics")


# =============================================================================
# Doctor (diagnostics)
# =============================================================================


def _run_doctor(test: bool = False) -> None:
    """Run system diagnostics."""
    from fitz_sage.core.detect import detect_system_status

    issues = []
    warnings = []

    # System
    ui.section("System")

    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    ui.status("Python", ok, f"Python {version}")
    if not ok:
        issues.append("Python 3.10+ required")

    workspace = FitzPaths.workspace()
    ui.status(
        "Workspace",
        workspace.exists(),
        str(workspace) if workspace.exists() else "Not found (run 'fitz init')",
    )
    if not workspace.exists():
        warnings.append("Run 'fitz init' to create workspace")

    ctx = CLIContext.load()
    ui.status(
        "Config",
        ctx.has_user_config,
        "Valid" if ctx.has_user_config else "Using defaults (run 'fitz init')",
    )
    if not ctx.has_user_config:
        warnings.append("Run 'fitz init' to create config")

    # Services
    ui.section("Services")

    system = detect_system_status()

    if system.llm_endpoint.available:
        ui.status(
            "OpenAI-compatible endpoint",
            True,
            f"{system.llm_endpoint.base_url} — {system.llm_endpoint.details}",
        )
    else:
        ui.warning("OpenAI-compatible endpoint", system.llm_endpoint.details)

    if system.pgvector.available:
        ui.status("pgvector", True, "installed")
    else:
        ui.warning("pgvector", system.pgvector.details)

    # API Keys
    ui.section("API Keys")

    for name, key_status in system.api_keys.items():
        if key_status.available:
            ui.status(name.capitalize(), True, "configured")
        else:
            ui.warning(name.capitalize(), f"${key_status.env_var} not set")

    # Connection Tests
    if test:
        ui.section("Connection Tests")

        # Embedding
        try:
            if ctx.embedding_plugin:
                embedder = ctx.get_embedder()
                vector = embedder.embed("test")
                if vector and len(vector) > 0:
                    ui.status("Embedding", True, f"{ctx.embedding_plugin} (dim={len(vector)})")
                else:
                    ui.status("Embedding", False, "Empty response")
                    issues.append("Embedding returned empty")
            else:
                ui.status("Embedding", False, "Not configured")
                issues.append("Embedding not configured")
        except Exception as e:
            ui.status("Embedding", False, str(e)[:50])
            issues.append(f"Embedding failed: {str(e)[:50]}")

        # Chat
        try:
            if ctx.chat_plugin:
                ctx.get_chat()
                ui.status("Chat", True, f"{ctx.chat_plugin} ready")
            else:
                ui.status("Chat", False, "Not configured")
                issues.append("Chat not configured")
        except Exception as e:
            ui.status("Chat", False, str(e)[:50])
            issues.append(f"Chat failed: {str(e)[:50]}")

        # Vector DB
        try:
            if ctx.vector_db_plugin:
                client = ctx.get_vector_db_client()
                collections = client.list_collections()
                ui.status(
                    "Vector DB", True, f"{ctx.vector_db_plugin} ({len(collections)} collections)"
                )
            else:
                ui.status("Vector DB", False, "Not configured")
                issues.append("Vector DB not configured")
        except Exception as e:
            ui.status("Vector DB", False, str(e)[:50])
            issues.append(f"Vector DB failed: {str(e)[:50]}")

    # Summary
    print()

    if issues:
        if RICH:
            console.print(
                Panel(
                    "\n".join(f"[red]x[/red] {issue}" for issue in issues),
                    title="[red]Issues Found[/red]",
                    border_style="red",
                )
            )
        else:
            print("Issues Found:")
            for issue in issues:
                print(f"  x {issue}")
        raise typer.Exit(1)
    elif warnings:
        ui.print("Some warnings, but Fitz should work.", "yellow")
        if not test:
            ui.info("Run 'fitz config --test' to verify connections")
    else:
        ui.success("All checks passed!")
        if not test:
            ui.info("Run 'fitz config --test' to verify connections")

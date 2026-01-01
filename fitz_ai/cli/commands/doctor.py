# fitz_ai/cli/commands/doctor.py
"""
System diagnostics command.

Usage:
    fitz doctor              # Run all checks
    fitz doctor --verbose    # Show more details
    fitz doctor --test       # Test actual connections
"""

from __future__ import annotations

import sys
from typing import Optional

import typer

from fitz_ai.cli.context import CLIContext
from fitz_ai.cli.ui import RICH, Panel, console, ui
from fitz_ai.core.detect import detect_system_status
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


# =============================================================================
# Check Functions
# =============================================================================


def _check_python() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version.split()[0]
    ok = sys.version_info >= (3, 10)
    return ok, f"Python {version}"


def _check_workspace() -> tuple[bool, str]:
    """Check if workspace exists."""
    workspace = FitzPaths.workspace()
    if workspace.exists():
        return True, str(workspace)
    return False, "Not found (run 'fitz init')"


def _check_config() -> tuple[bool, str, Optional[CLIContext]]:
    """Check if config exists and is valid."""
    ctx = CLIContext.load_or_none()

    if ctx is None:
        return False, "Not found (run 'fitz init')", None

    return True, "Valid", ctx


def _check_dependencies() -> list[tuple[str, bool, str]]:
    """Check required Python packages."""
    packages = [
        ("typer", "CLI framework"),
        ("httpx", "HTTP client"),
        ("pydantic", "Config validation"),
        ("yaml", "YAML parsing", "pyyaml"),
        ("rich", "Pretty output", "rich"),
    ]

    results = []
    for item in packages:
        name = item[0]
        item[1]
        pip_pkg = item[2] if len(item) > 2 else item[0]

        try:
            # FIX: Import using 'name' (the module name), not 'pip_pkg'
            mod = __import__(name.replace("-", "_"))
            version = getattr(mod, "__version__", "")
            results.append((name, True, version))
        except ImportError:
            results.append((name, False, f"pip install {pip_pkg}"))

    return results


def _check_optional_dependencies() -> list[tuple[str, bool, str]]:
    """Check optional Python packages."""
    packages = [
        ("faiss", "Local vector DB", "faiss-cpu"),
        ("qdrant_client", "Qdrant client", "qdrant-client"),
        ("cohere", "Cohere API", "cohere"),
        ("openai", "OpenAI API", "openai"),
    ]

    results = []
    for item in packages:
        name = item[0]
        desc = item[1]
        item[2] if len(item) > 2 else item[0]

        try:
            mod = __import__(name.replace("-", "_"))
            version = getattr(mod, "__version__", "installed")
            results.append((desc, True, version))
        except ImportError:
            results.append((desc, False, "not installed"))

    return results


def _test_embedding(ctx: CLIContext) -> tuple[bool, str]:
    """Test embedding plugin."""
    try:
        from fitz_ai.llm.registry import get_llm_plugin

        if not ctx.embedding_plugin:
            return False, "Not configured"

        plugin = get_llm_plugin(plugin_type="embedding", plugin_name=ctx.embedding_plugin)

        # Try a test embedding
        vector = plugin.embed("test")
        if vector and len(vector) > 0:
            return True, f"{ctx.embedding_plugin} (dim={len(vector)})"
        return False, "Empty response"

    except Exception as e:
        return False, str(e)[:50]


def _test_chat(ctx: CLIContext) -> tuple[bool, str]:
    """Test chat plugin (without making actual call)."""
    try:
        from fitz_ai.llm.registry import get_llm_plugin

        if not ctx.chat_plugin:
            return False, "Not configured"

        # Just try to instantiate
        get_llm_plugin(plugin_type="chat", plugin_name=ctx.chat_plugin)
        return True, f"{ctx.chat_plugin} ready"

    except Exception as e:
        return False, str(e)[:50]


def _test_vector_db(ctx: CLIContext) -> tuple[bool, str]:
    """Test vector DB connection."""
    try:
        if not ctx.vector_db_plugin:
            return False, "Not configured"

        client = ctx.get_vector_db_client()

        # Try to list collections
        collections = client.list_collections()
        return True, f"{ctx.vector_db_plugin} ({len(collections)} collections)"

    except Exception as e:
        return False, str(e)[:50]


def _test_rerank(ctx: CLIContext) -> tuple[bool, str]:
    """Test rerank plugin."""
    if not ctx.rerank_enabled:
        return True, "Disabled (optional)"

    try:
        from fitz_ai.llm.registry import get_llm_plugin

        if not ctx.rerank_plugin:
            return False, "Enabled but no plugin"

        # Just try to instantiate
        get_llm_plugin(plugin_type="rerank", plugin_name=ctx.rerank_plugin)
        return True, f"{ctx.rerank_plugin} ready"

    except Exception as e:
        return False, str(e)[:50]


# =============================================================================
# Main Command
# =============================================================================


def command(
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output.",
    ),
    test: bool = typer.Option(
        False,
        "--test",
        "-t",
        help="Test actual connections.",
    ),
) -> None:
    """
    Run diagnostics on your Fitz setup.

    Checks system requirements, configuration, and connections.

    Examples:
        fitz doctor           # Quick check
        fitz doctor -v        # Verbose output
        fitz doctor --test    # Test actual connections
    """
    issues = []
    warnings = []

    # =========================================================================
    # Header
    # =========================================================================

    ui.header("Fitz Doctor", "System diagnostics")

    # =========================================================================
    # Core Checks
    # =========================================================================

    ui.section("System")

    # Python
    ok, detail = _check_python()
    ui.status("Python", ok, detail)
    if not ok:
        issues.append("Python 3.10+ required")

    # Workspace
    ok, detail = _check_workspace()
    ui.status("Workspace", ok, detail)
    if not ok:
        warnings.append("Run 'fitz init' to create workspace")

    # Config
    ok, detail, ctx = _check_config()
    ui.status("Config", ok, detail)
    if not ok:
        warnings.append("Run 'fitz init' to create config")

    # =========================================================================
    # Dependencies
    # =========================================================================

    ui.section("Dependencies")

    for name, ok, detail in _check_dependencies():
        ui.status(name, ok, detail)
        if not ok:
            issues.append(f"Missing: {name}")

    # =========================================================================
    # Optional Dependencies (verbose only)
    # =========================================================================

    if verbose:
        ui.section("Optional Packages")
        for name, ok, detail in _check_optional_dependencies():
            if ok:
                ui.status(name, True, detail)
            else:
                ui.warning(name, detail)

    # =========================================================================
    # Services
    # =========================================================================

    ui.section("Services")

    system = detect_system_status()

    # Ollama
    if system.ollama.available:
        ui.status("Ollama", True, system.ollama.details)
    else:
        ui.warning("Ollama", system.ollama.details)

    # Qdrant
    if system.qdrant.available:
        ui.status("Qdrant", True, f"{system.qdrant.host}:{system.qdrant.port}")
    else:
        ui.warning("Qdrant", system.qdrant.details)

    # FAISS
    if system.faiss.available:
        ui.status("FAISS", True, "installed")
    else:
        ui.warning("FAISS", system.faiss.details)

    # =========================================================================
    # API Keys
    # =========================================================================

    ui.section("API Keys")

    for name, key_status in system.api_keys.items():
        if key_status.available:
            ui.status(name.capitalize(), True, "configured")
        else:
            ui.warning(name.capitalize(), f"${key_status.env_var} not set")

    # =========================================================================
    # Connection Tests (if --test)
    # =========================================================================

    if test and ctx:
        ui.section("Connection Tests")

        # Embedding
        ok, detail = _test_embedding(ctx)
        ui.status("Embedding", ok, detail)
        if not ok:
            issues.append(f"Embedding failed: {detail}")

        # Chat
        ok, detail = _test_chat(ctx)
        ui.status("Chat", ok, detail)
        if not ok:
            issues.append(f"Chat failed: {detail}")

        # Vector DB
        ok, detail = _test_vector_db(ctx)
        ui.status("Vector DB", ok, detail)
        if not ok:
            issues.append(f"Vector DB failed: {detail}")

        # Rerank
        ok, detail = _test_rerank(ctx)
        ui.status("Rerank", ok, detail)

    # =========================================================================
    # Config Summary (verbose)
    # =========================================================================

    if verbose and ctx:
        ui.section("Config Summary")

        rerank_str = ctx.rerank_plugin or "disabled" if ctx.rerank_enabled else "disabled"

        ui.info(f"Chat: {ctx.chat_plugin or '?'}")
        ui.info(f"Embedding: {ctx.embedding_plugin or '?'}")
        ui.info(f"Vector DB: {ctx.vector_db_plugin or '?'}")
        ui.info(f"Rerank: {rerank_str}")

    # =========================================================================
    # Summary
    # =========================================================================

    print()

    if issues:
        if RICH:
            console.print(
                Panel(
                    "\n".join(f"[red]✗[/red] {issue}" for issue in issues),
                    title="[red]Issues Found[/red]",
                    border_style="red",
                )
            )
        else:
            print("Issues Found:")
            for issue in issues:
                print(f"  ✗ {issue}")

        raise typer.Exit(1)

    elif warnings:
        ui.print("Some warnings, but Fitz should work.", "yellow")

        if not test:
            ui.info("Run 'fitz doctor --test' to verify connections")

    else:
        ui.success("All checks passed!")

        if not test:
            ui.info("Run 'fitz doctor --test' to verify connections")

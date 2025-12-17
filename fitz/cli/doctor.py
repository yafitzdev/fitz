# fitz/cli/doctor.py
"""
Doctor command for Fitz CLI.

Runs comprehensive diagnostics on your Fitz setup.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import typer

# Import centralized detection
from fitz.core.detect import (
    ProviderStatus,
    detect_all,
    detect_api_key,
)

# Rich for pretty output (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_status(name: str, status: str, details: str = "", ok: bool = True) -> None:
    """Print a status line."""
    if RICH_AVAILABLE:
        icon = "✓" if ok else "✗" if status == "ERROR" else "⚠"
        color = "green" if ok else "red" if status == "ERROR" else "yellow"
        console.print(f"  [{color}]{icon}[/{color}] {name}: {details}")
    else:
        icon = "✓" if ok else "✗" if status == "ERROR" else "⚠"
        print(f"  {icon} {name}: {details}")


def print_provider_status(provider: ProviderStatus) -> None:
    """Print status for a provider."""
    status = "OK" if provider.available else "INFO"
    print_status(provider.name, status, provider.details, provider.available)


def check_python() -> tuple[bool, str]:
    """Check Python version."""
    version = sys.version.split()[0]
    major, minor = sys.version_info[:2]
    ok = major >= 3 and minor >= 10
    return ok, f"Python {version}" + ("" if ok else " (3.10+ required)")


def check_fitz_dir() -> tuple[bool, str]:
    """Check for .fitz directory."""
    fitz_dir = Path.cwd() / ".fitz"
    if fitz_dir.exists():
        config = fitz_dir / "config.yaml"
        if config.exists():
            return True, ".fitz/config.yaml exists"
        return True, ".fitz/ exists (no config.yaml)"
    return False, ".fitz/ not found (run: fitz init)"


def check_config() -> tuple[bool, str, Optional[dict]]:
    """Check if config is valid."""
    config_paths = [
        Path.cwd() / ".fitz" / "config.yaml",
        Path.cwd() / "fitz" / "engines" / "classic_rag" / "config" / "default.yaml",
    ]

    for path in config_paths:
        if path.exists():
            try:
                import yaml

                with open(path) as f:
                    config = yaml.safe_load(f)
                return True, f"Config loaded from {path.name}", config
            except Exception as e:
                return False, f"Config error: {e}", None

    return False, "No config found", None


def check_dependencies() -> list[tuple[str, bool, str]]:
    """Check required Python dependencies."""
    deps = []

    packages = [
        ("typer", "CLI framework"),
        ("httpx", "HTTP client"),
        ("pydantic", "Config validation"),
        ("yaml", "YAML parsing", "pyyaml"),
    ]

    for item in packages:
        name = item[0]
        desc = item[1]
        import_name = item[2] if len(item) > 2 else item[0]

        try:
            __import__(import_name.replace("-", "_"))
            deps.append((f"{name}", True, "Installed"))
        except ImportError:
            deps.append((f"{name}", False, f"Missing (pip install {import_name})"))

    return deps


def test_chat(config: dict) -> tuple[bool, str]:
    """Test chat/LLM configuration."""
    llm_cfg = config.get("llm", {})
    plugin = llm_cfg.get("plugin_name", "?")
    model = llm_cfg.get("kwargs", {}).get("model", "default")
    return True, f"Plugin '{plugin}' model '{model}'"


def test_embedding(config: dict) -> tuple[bool, str]:
    """Test embedding configuration."""
    emb_cfg = config.get("embedding", {})
    plugin = emb_cfg.get("plugin_name", "?")
    model = emb_cfg.get("kwargs", {}).get("model", "default")
    return True, f"Plugin '{plugin}'" + (
        f" model '{model}'" if model != "default" else " configured"
    )


def test_rerank(config: dict) -> tuple[bool, str]:
    """Test rerank configuration."""
    rerank_cfg = config.get("rerank", {})

    # Check if rerank is configured
    if not rerank_cfg:
        return False, "Not configured (optional)"

    enabled = rerank_cfg.get("enabled", False)
    if not enabled:
        return False, "Disabled (optional)"

    plugin = rerank_cfg.get("plugin_name", "?")
    model = rerank_cfg.get("kwargs", {}).get("model", "")

    if plugin == "?" or plugin is None:
        return False, "Enabled but no plugin specified"

    detail = f"Plugin '{plugin}'"
    if model:
        detail += f" model '{model}'"

    return True, detail


def command(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show detailed output"),
    test_connections: bool = typer.Option(False, "--test", "-t", help="Test actual connections"),
) -> None:
    """
    Run diagnostics on your Fitz setup.

    Checks:
    - Python version and dependencies
    - Configuration files
    - API keys and credentials
    - External services (Ollama, Qdrant)

    Examples:
        fitz doctor           # Quick check
        fitz doctor -v        # Verbose output
        fitz doctor --test    # Test actual connections
    """

    if RICH_AVAILABLE:
        console.print(
            Panel.fit("[bold]🩺 Fitz Doctor[/bold]\n" "Running diagnostics...", border_style="blue")
        )
    else:
        print("\n" + "=" * 60)
        print("🩺 Fitz Doctor")
        print("Running diagnostics...")
        print("=" * 60)

    issues = []
    warnings = []

    # =========================================================================
    # Core Checks
    # =========================================================================

    print("\n📦 Core:")

    ok, msg = check_python()
    print_status("Python", "OK" if ok else "ERROR", msg, ok)
    if not ok:
        issues.append("Python 3.10+ required")

    ok, msg = check_fitz_dir()
    print_status("Workspace", "OK" if ok else "WARN", msg, ok)
    if not ok:
        warnings.append("Run 'fitz init' to create workspace")

    ok, msg, config = check_config()
    print_status("Config", "OK" if ok else "WARN", msg, ok)
    if not ok:
        warnings.append("Run 'fitz init' to create config")

    # =========================================================================
    # Dependencies
    # =========================================================================

    if verbose:
        print("\n📚 Dependencies:")
        for name, ok, msg in check_dependencies():
            print_status(name, "OK" if ok else "ERROR", msg, ok)
            if not ok:
                issues.append(f"Missing dependency: {name}")

    # =========================================================================
    # LLM Providers (using centralized detection)
    # =========================================================================

    print("\n🤖 LLM Providers:")

    system = detect_all()

    print_provider_status(system.ollama)
    print_provider_status(system.api_keys["cohere"])
    print_provider_status(system.api_keys["openai"])
    print_provider_status(system.api_keys["anthropic"])

    # =========================================================================
    # Vector Databases (using centralized detection)
    # =========================================================================

    print("\n🗄️  Vector Databases:")

    print_provider_status(system.qdrant)
    print_provider_status(system.faiss)

    # =========================================================================
    # Config Validation
    # =========================================================================

    if config:
        print("\n⚙️  Configuration:")

        # Chat (was "LLM")
        ok, msg = test_chat(config)
        print_status("Chat", "OK" if ok else "WARN", msg, ok)
        if not ok:
            warnings.append(f"Chat: {msg}")

        # Embedding
        ok, msg = test_embedding(config)
        print_status("Embedding", "OK" if ok else "WARN", msg, ok)
        if not ok:
            warnings.append(f"Embedding: {msg}")

        # Rerank (NEW)
        ok, msg = test_rerank(config)
        print_status("Rerank", "OK" if ok else "INFO", msg, ok)

        # Show configured values in verbose mode
        if verbose:
            print("\n  Configured values:")
            print(f"    Chat: {config.get('llm', {}).get('plugin_name', '?')}")
            print(f"    Embedding: {config.get('embedding', {}).get('plugin_name', '?')}")
            print(f"    Rerank: {config.get('rerank', {}).get('plugin_name', 'disabled')}")
            print(f"    Vector DB: {config.get('vector_db', {}).get('plugin_name', '?')}")
            print(f"    Collection: {config.get('retriever', {}).get('collection', '?')}")

    # =========================================================================
    # Connection Tests
    # =========================================================================

    if test_connections and config:
        print("\n🔌 Connection Tests:")

        # Test Qdrant connection with actual client
        try:
            from qdrant_client import QdrantClient

            vdb_cfg = config.get("vector_db", {}).get("kwargs", {})
            # Use detected host/port as fallback
            host = vdb_cfg.get("host", system.qdrant_host)
            port = vdb_cfg.get("port", system.qdrant_port)
            client = QdrantClient(host=host, port=port, timeout=5)
            collections = client.get_collections()
            print_status(
                "Qdrant connection",
                "OK",
                f"Connected, {len(collections.collections)} collections",
                True,
            )
        except Exception as e:
            print_status("Qdrant connection", "ERROR", str(e), False)
            issues.append(f"Qdrant connection failed: {e}")

        # Test embedding
        try:
            # This would need actual embedding test implementation
            print_status("Embedding test", "SKIP", "Not implemented", True)
        except Exception as e:
            print_status("Embedding test", "ERROR", str(e), False)

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)

    if issues:
        if RICH_AVAILABLE:
            console.print(f"[red]❌ {len(issues)} issue(s) found:[/red]")
            for issue in issues:
                console.print(f"  [red]•[/red] {issue}")
        else:
            print(f"❌ {len(issues)} issue(s) found:")
            for issue in issues:
                print(f"  • {issue}")
    elif warnings:
        if RICH_AVAILABLE:
            console.print(f"[yellow]⚠️  {len(warnings)} warning(s):[/yellow]")
            for warning in warnings:
                console.print(f"  [yellow]•[/yellow] {warning}")
        else:
            print(f"⚠️  {len(warnings)} warning(s):")
            for warning in warnings:
                print(f"  • {warning}")
        print("✅ All checks passed!")
    else:
        print("✅ All checks passed!")


if __name__ == "__main__":
    typer.run(command)

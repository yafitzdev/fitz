# fitz/cli/doctor.py
"""
Doctor command: Run diagnostics on Fitz setup.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from fitz_ai.core.detect import detect_all

# Rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def print_status(name: str, status: str, details: str, ok: bool) -> None:
    """Print a status line."""
    if RICH_AVAILABLE:
        icon = "✓" if ok else "⚠" if status == "WARN" or status == "INFO" else "✗"
        color = "green" if ok else "yellow" if status == "WARN" or status == "INFO" else "red"
        console.print(f"  [{color}]{icon}[/{color}] {name}: {details}")
    else:
        icon = "✓" if ok else "⚠" if status == "WARN" else "✗"
        print(f"  {icon} {name}: {details}")


def print_provider_status(provider) -> None:
    """Print status for a provider."""
    print_status(
        provider.name, "OK" if provider.available else "WARN", provider.details, provider.available
    )


def check_python() -> tuple[bool, str]:
    """Check Python version."""
    import sys

    version = f"Python {sys.version.split()[0]}"
    ok = sys.version_info >= (3, 10)
    return ok, version


def check_fitz_dir() -> tuple[bool, str]:
    """Check if .fitz directory exists."""
    fitz_dir = Path.cwd() / ".fitz"
    config_file = fitz_dir / "config.yaml"

    if config_file.exists():
        return True, ".fitz/config.yaml exists"
    elif fitz_dir.exists():
        return True, ".fitz/ exists (no config.yaml)"
    else:
        return False, "Not found (run 'fitz init')"


def check_config() -> tuple[bool, str, Optional[dict]]:
    """Check if config can be loaded."""
    try:
        from fitz_ai.engines.classic_rag.config.loader import load_config_dict

        config = load_config_dict()
        return True, "Config loaded from config.yaml", config
    except Exception as e:
        return False, f"Failed to load: {e}", None


def check_dependencies() -> list[tuple[str, bool, str]]:
    """Check required dependencies."""
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
    """Test chat configuration."""
    chat_cfg = config.get("chat", {})
    plugin = chat_cfg.get("plugin_name", "?")
    model = chat_cfg.get("kwargs", {}).get("model", "default")

    if plugin == "?":
        return False, "Not configured (missing 'chat' in config)"

    return True, f"Plugin '{plugin}' model '{model}'"


def test_embedding(config: dict) -> tuple[bool, str]:
    """Test embedding configuration."""
    emb_cfg = config.get("embedding", {})
    plugin = emb_cfg.get("plugin_name", "?")
    model = emb_cfg.get("kwargs", {}).get("model", "default")

    if plugin == "?":
        return False, "Not configured"

    return True, f"Plugin '{plugin}'" + (
        f" model '{model}'" if model != "default" else " configured"
    )


def test_rerank(config: dict) -> tuple[bool, str]:
    """Test rerank configuration."""
    rerank_cfg = config.get("rerank", {})

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

        # Chat
        ok, msg = test_chat(config)
        print_status("Chat", "OK" if ok else "WARN", msg, ok)
        if not ok:
            warnings.append(f"Chat: {msg}")

        # Embedding
        ok, msg = test_embedding(config)
        print_status("Embedding", "OK" if ok else "WARN", msg, ok)
        if not ok:
            warnings.append(f"Embedding: {msg}")

        # Rerank
        ok, msg = test_rerank(config)
        print_status("Rerank", "OK" if ok else "INFO", msg, ok)

        # Show configured values in verbose mode
        if verbose:
            print("\n  Configured values:")
            print(f"    Chat: {config.get('chat', {}).get('plugin_name', '?')}")
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

# fitz/cli/cli.py

"""
Main Fitz CLI.

Goals:
- Discoverability first
- Zero magic
- No core side effects
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import typer

from fitz.core.llm.registry import LLM_REGISTRY
from fitz.pipeline.pipeline.registry import available_pipeline_plugins
from fitz.ingest.chunking.registry import CHUNKER_REGISTRY
from fitz.ingest.ingestion.registry import REGISTRY as INGEST_REGISTRY

app = typer.Typer(
    help="Fitz — local-first RAG framework",
    no_args_is_help=True,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def check_ollama_installed() -> bool:
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


# ---------------------------------------------------------------------------
# help / overview
# ---------------------------------------------------------------------------

@app.command("help")
def help_cmd() -> None:
    typer.echo()
    typer.echo("Fitz — local-first RAG framework")
    typer.echo()
    typer.echo("Common commands:")
    typer.echo("  fitz init               Initialize a local Fitz workspace")
    typer.echo("  fitz plugins            List available plugins")
    typer.echo("  fitz ingest             Ingest documents")
    typer.echo("  fitz pipeline run       Query your knowledge base")
    typer.echo("  fitz config show        Show resolved configuration")
    typer.echo()
    typer.echo("Quick start:")
    typer.echo("  1. fitz init")
    typer.echo("  2. fitz ingest ./docs")
    typer.echo('  3. fitz pipeline run "What is this project about?"')
    typer.echo()


# ---------------------------------------------------------------------------
# init
# ---------------------------------------------------------------------------

@app.command("init")
def init() -> None:
    """
    Initialize a local Fitz workspace.

    Creates .fitz/ if it does not exist and prints
    the active default setup.
    """
    root = Path.cwd()
    fitz_dir = root / ".fitz"

    if not fitz_dir.exists():
        fitz_dir.mkdir()
        typer.echo("✓ Created .fitz/")
    else:
        typer.echo("✓ .fitz/ already exists")

    typer.echo()
    typer.echo("Active defaults:")
    typer.echo("  Preset: local")
    typer.echo("  LLM: local (Ollama)")
    typer.echo("  Embedding: local")
    typer.echo("  Vector DB: local-faiss (disk)")
    typer.echo()
    typer.echo("Next steps:")
    typer.echo("  fitz ingest ./docs")
    typer.echo('  fitz pipeline run "Your question"')
    typer.echo()


# ---------------------------------------------------------------------------
# plugins
# ---------------------------------------------------------------------------

def _discover_all_plugins() -> None:
    # LLM
    import fitz.core.llm.chat.plugins  # noqa
    import fitz.core.llm.embedding.plugins  # noqa
    import fitz.core.llm.rerank.plugins  # noqa

    # Vector DB
    import fitz.core.vector_db.plugins  # noqa


@app.command("plugins")
def plugins() -> None:
    """
    List all discovered plugins.
    """
    typer.echo()

    # Trigger discovery explicitly
    _discover_all_plugins()

    from fitz.core.llm.registry import LLM_REGISTRY

    def show(title: str, plugins: dict):
        typer.echo(f"{title}:")
        if not plugins:
            typer.echo("  (none)")
        else:
            for name in sorted(plugins):
                typer.echo(f"  - {name}")
        typer.echo()

    show("LLM chat", LLM_REGISTRY.get("chat", {}))
    show("LLM embedding", LLM_REGISTRY.get("embedding", {}))
    show("LLM rerank", LLM_REGISTRY.get("rerank", {}))
    show("Vector DB", LLM_REGISTRY.get("vector_db", {}))


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------

@app.command("doctor")
def doctor() -> None:
    """
    Run basic diagnostics for local setup.
    """
    typer.echo()
    typer.echo("Fitz doctor")
    typer.echo()

    # Python
    typer.echo(f"✓ Python {sys.version.split()[0]}")

    # .fitz
    if Path(".fitz").exists():
        typer.echo("✓ .fitz directory present")
    else:
        typer.echo("⚠ .fitz directory missing (run: fitz init)")

    # Ollama
    if check_ollama_installed():
        typer.echo("✓ Ollama detected")
    else:
        typer.echo("⚠ Ollama not detected (local LLM unavailable)")

    typer.echo()


# ---------------------------------------------------------------------------
# legacy local-llm helpers (kept for compatibility)
# ---------------------------------------------------------------------------

@app.command("setup-local")
def setup_local() -> None:
    typer.echo()
    typer.echo("Deprecated: use `fitz init` + `fitz doctor`")
    typer.echo("This command will be removed in a future release.")
    typer.echo()


@app.command("test")
def test() -> None:
    typer.echo()
    typer.echo("Deprecated: use `fitz doctor`")
    typer.echo("This command will be removed in a future release.")
    typer.echo()


if __name__ == "__main__":
    app()

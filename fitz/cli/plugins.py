"""Plugins command for Fitz CLI."""

import typer


def command() -> None:
    """
    List all discovered plugins.
    """
    from fitz.core.llm.registry import available_llm_plugins

    typer.echo()

    def show(title: str, plugin_type: str):
        typer.echo(f"{title}:")
        names = available_llm_plugins(plugin_type)
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")
        typer.echo()

    show("LLM chat", "chat")
    show("LLM embedding", "embedding")
    show("LLM rerank", "rerank")

    # Vector DB plugins
    show("Vector DB", "vector_db")
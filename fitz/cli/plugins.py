# fitz/cli/plugins.py
"""
Plugins command for Fitz CLI.

Lists all discovered plugins across all registries.
"""

import typer


def command() -> None:
    """
    List all discovered plugins.
    """
    from fitz.core.registry import (
        available_llm_plugins,
        available_vector_db_plugins,
    )

    typer.echo()

    def show_llm(title: str, plugin_type: str):
        """Show LLM plugins (chat, embedding, rerank)."""
        typer.echo(f"{title}:")
        names = available_llm_plugins(plugin_type)
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")
        typer.echo()

    def show_vector_db(title: str):
        """Show Vector DB plugins (separate registry)."""
        typer.echo(f"{title}:")
        names = available_vector_db_plugins()
        if not names:
            typer.echo("  (none)")
        else:
            for name in sorted(names):
                typer.echo(f"  - {name}")
        typer.echo()

    # LLM plugins
    show_llm("LLM Chat", "chat")
    show_llm("LLM Embedding", "embedding")
    show_llm("LLM Rerank", "rerank")

    # Vector DB plugins (separate registry!)
    show_vector_db("Vector DB")

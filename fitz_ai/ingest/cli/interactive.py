# fitz_ai/ingest/cli/interactive.py
"""
Interactive ingestion mode - simple and clean.

Usage:
    fitz ingest  # No arguments triggers interactive mode
"""

from pathlib import Path
from typing import Optional

import questionary
from rich.console import Console
from rich.panel import Panel

console = Console()


def run_interactive_ingestion() -> Optional[dict]:
    """
    Run interactive ingestion flow.

    Flow:
    1. Enter path
    2. Select chunker (1, 2, 3)
    3. Enter collection name (optional)
    4. Show config info (no input)
    5. Confirm

    Returns:
        Config dict or None if cancelled
    """

    console.print(Panel.fit(
        "[bold]🔄 Interactive Ingestion[/bold]",
        border_style="blue"
    ))
    console.print()

    # Step 1: Path
    path_str = questionary.path(
        "Enter path:",
        validate=lambda p: Path(p).exists() or "Path does not exist"
    ).ask()

    if not path_str:
        return None

    source = Path(path_str)

    # Show file info
    if source.is_file():
        size_mb = source.stat().st_size / (1024 * 1024)
        console.print(f"\n✓ Found: {source.name} ({size_mb:.1f} MB)\n")
    else:
        num_files = len(list(source.rglob("*")))
        console.print(f"\n✓ Found: {source.name} ({num_files} files)\n")

    # Step 2: Chunker selection
    console.print("[bold]Select chunker:[/bold]")

    from fitz_ai.core.registry import available_chunking_plugins
    available = available_chunking_plugins()

    # Clean descriptions
    chunker_descriptions = {
        'simple': 'Fixed-size chunks',
        'pdf_sections': 'Section-based chunking',
        'semantic': 'Semantic similarity',
    }

    choices = []
    for i, name in enumerate(available, 1):
        desc = chunker_descriptions.get(name, 'Custom chunker')
        console.print(f"  {i}. {name} - {desc}")
        choices.append(name)

    console.print()

    # Get selection
    choice = questionary.text(
        "Choice:",
        validate=lambda x: (x.isdigit() and 1 <= int(x) <= len(choices)) or "Enter a number"
    ).ask()

    if not choice:
        return None

    chunker = choices[int(choice) - 1]

    console.print(f"\n✓ Using: [bold]{chunker}[/bold]\n")
    console.print("─" * 50)
    console.print()

    # Step 3: Collection name
    collection = questionary.text(
        "Collection name [default]:",
        default=""
    ).ask()

    if collection is None:
        return None

    collection = collection or "default"

    console.print()
    console.print("─" * 50)
    console.print()

    # Step 4: Show config (info only, no input)
    console.print("[bold]Configuration[/bold] (from ~/.config/fitz/config.yaml):")

    try:
        from fitz_ai.core.config import load_rag_config
        config = load_rag_config()

        embedding = config.get('embedding', {}).get('plugin_name', 'cohere')
        vector_db = config.get('vector_db', {}).get('plugin_name', 'qdrant')
        rerank_enabled = config.get('rerank', {}).get('enabled', False)

        console.print(f"  • Embedding: {embedding}")
        console.print(f"  • Vector DB: {vector_db}")
        console.print(f"  • Reranker: {'enabled' if rerank_enabled else 'disabled'}")
    except Exception:
        # Fallback to defaults if config can't be loaded
        console.print("  • Embedding: cohere")
        console.print("  • Vector DB: qdrant")
        console.print("  • Reranker: disabled")

    console.print()
    console.print("─" * 50)
    console.print()

    # Step 5: Confirmation
    proceed = questionary.confirm(
        "Proceed?",
        default=True
    ).ask()

    if not proceed:
        console.print("\n[yellow]Cancelled.[/yellow]")
        return None

    console.print()

    return {
        'source': source,
        'collection': collection,
        'chunker': chunker,
    }


def integrate_with_run_command(config: dict):
    """
    Call the run command with config from interactive mode.

    Args:
        config: Dict with source, collection, chunker
    """
    from fitz_ai.ingest.cli.run import command as run_command

    # Get embedding/vector_db from actual config if possible
    try:
        from fitz_ai.core.config import load_rag_config
        rag_config = load_rag_config()
        embedding_plugin = rag_config.get('embedding', {}).get('plugin_name', 'cohere')
        vector_db_plugin = rag_config.get('vector_db', {}).get('plugin_name', 'qdrant')
    except Exception:
        # Fallback to defaults
        embedding_plugin = 'cohere'
        vector_db_plugin = 'qdrant'

    # Call run command with interactive config
    run_command(
        source=config['source'],
        collection=config['collection'],
        ingest_plugin='local',
        chunker=config['chunker'],
        chunk_size=1000,  # Use defaults
        chunk_overlap=0,
        min_section_chars=50,
        max_section_chars=3000,
        embedding_plugin=embedding_plugin,
        vector_db_plugin=vector_db_plugin,
        batch_size=50,
        quiet=False,
    )
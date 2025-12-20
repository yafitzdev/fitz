# fitz_ai/cli/v2/commands/ingest.py
"""
Interactive document ingestion.

Usage:
    fitz ingest              # Interactive mode - prompts for everything
    fitz ingest ./docs       # Skip source prompt
    fitz ingest -y           # Use defaults, no prompts
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Optional

import typer

from fitz_ai.core.config import load_config_dict, ConfigNotFoundError
from fitz_ai.core.paths import FitzPaths
from fitz_ai.core.registry import available_chunking_plugins
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.ingest.chunking.engine import ChunkingEngine
from fitz_ai.ingest.config.schema import ChunkerConfig
from fitz_ai.ingest.ingestion.engine import IngestionEngine
from fitz_ai.ingest.ingestion.registry import get_ingest_plugin
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.vector_db.registry import get_vector_db_plugin
from fitz_ai.vector_db.writer import VectorDBWriter
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Rich for UI (optional)
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.prompt import Prompt, Confirm, IntPrompt
    from rich.progress import (
        Progress,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TaskProgressColumn,
        TimeElapsedColumn,
    )

    console = Console()
    RICH = True
except ImportError:
    console = None
    RICH = False


# =============================================================================
# UI Helpers
# =============================================================================


def _print(msg: str, style: str = "") -> None:
    if RICH and style:
        console.print(f"[{style}]{msg}[/{style}]")
    else:
        print(msg)


def _header(title: str) -> None:
    if RICH:
        console.print(Panel.fit(f"[bold]{title}[/bold]", border_style="blue"))
    else:
        print(f"\n{'=' * 50}")
        print(title)
        print('=' * 50)


def _step(num: int, total: int, msg: str) -> None:
    if RICH:
        console.print(f"[bold blue][{num}/{total}][/bold blue] {msg}")
    else:
        print(f"[{num}/{total}] {msg}")


def _success(msg: str) -> None:
    if RICH:
        console.print(f"[green]✓[/green] {msg}")
    else:
        print(f"✓ {msg}")


def _error(msg: str) -> None:
    if RICH:
        console.print(f"[red]✗[/red] {msg}")
    else:
        print(f"✗ {msg}")


def _prompt_text(prompt: str, default: str) -> str:
    if RICH:
        return Prompt.ask(prompt, default=default)
    else:
        response = input(f"{prompt} [{default}]: ").strip()
        return response if response else default


def _prompt_int(prompt: str, default: int) -> int:
    if RICH:
        return IntPrompt.ask(prompt, default=default)
    else:
        response = input(f"{prompt} [{default}]: ").strip()
        return int(response) if response else default


def _prompt_choice(prompt: str, choices: list[str], default: str) -> str:
    if RICH:
        return Prompt.ask(prompt, choices=choices, default=default)
    else:
        choices_str = "/".join(choices)
        while True:
            response = input(f"{prompt} [{choices_str}] ({default}): ").strip()
            if not response:
                return default
            if response in choices:
                return response
            print(f"Choose from: {', '.join(choices)}")


def _prompt_path(prompt: str, default: str = ".") -> Path:
    """Prompt for a path with validation."""
    while True:
        if RICH:
            path_str = Prompt.ask(prompt, default=default)
        else:
            response = input(f"{prompt} [{default}]: ").strip()
            path_str = response if response else default

        path = Path(path_str).expanduser().resolve()

        if path.exists():
            return path
        else:
            _error(f"Path does not exist: {path}")
            if RICH:
                if not Confirm.ask("Try again?", default=True):
                    raise typer.Exit(1)
            else:
                retry = input("Try again? [Y/n]: ").strip().lower()
                if retry in ("n", "no"):
                    raise typer.Exit(1)


# =============================================================================
# Config Loading
# =============================================================================


def _load_config() -> dict:
    """Load config or exit with helpful message."""
    try:
        return load_config_dict(FitzPaths.config())
    except ConfigNotFoundError:
        _error("No config found. Run 'fitz init' first.")
        raise typer.Exit(1)
    except Exception as e:
        _error(f"Failed to load config: {e}")
        raise typer.Exit(1)


# =============================================================================
# Batch Embedding with Progress
# =============================================================================


def _has_batch_embed(embedder: Any) -> bool:
    """Check if embedder supports batch embedding."""
    return hasattr(embedder, 'embed_batch') or hasattr(embedder, 'embed_texts')


def _embed_batch_native(embedder: Any, texts: List[str]) -> List[List[float]]:
    """Call native batch embed if available."""
    if hasattr(embedder, 'embed_batch'):
        return embedder.embed_batch(texts)
    elif hasattr(embedder, 'embed_texts'):
        return embedder.embed_texts(texts)
    else:
        raise NotImplementedError("No batch embed method")


def _is_batch_size_error(error: Exception) -> bool:
    """Check if error is due to batch size being too large."""
    error_str = str(error).lower()
    indicators = [
        "too many",
        "batch size",
        "maximum",
        "limit",
        "exceeded",
        "too large",
        "rate limit",
        "413",  # Payload too large
        "400",  # Bad request (often batch size)
    ]
    return any(ind in error_str for ind in indicators)


def _embed_chunks(
        embedder: Any,
        chunks: List[Chunk],
        batch_size: int = 96,
        show_progress: bool = True,
) -> List[List[float]]:
    """
    Embed chunks with adaptive batching and progress bar.

    Batching significantly reduces API calls:
    - 1000 chunks with batch_size=96 = ~11 API calls instead of 1000

    Adaptive batch sizing:
    - Starts at batch_size (default 96)
    - If batch fails, halves batch size and retries
    - Continues halving until batch_size=1 (single embedding)
    - Remembers working batch size for remaining chunks

    Args:
        embedder: Embedding plugin with embed() method
        chunks: List of chunks to embed
        batch_size: Initial batch size (will adapt down if needed)
        show_progress: Whether to show progress bar

    Returns:
        List of embedding vectors
    """
    vectors: List[List[float]] = []
    total_chunks = len(chunks)

    # Check if embedder has native batch support
    has_native_batch = _has_batch_embed(embedder)

    if not has_native_batch:
        # Fall back to single embedding
        return _embed_chunks_sequential(embedder, chunks, show_progress)

    # Adaptive batch embedding
    current_batch_size = batch_size
    position = 0

    if RICH and show_progress:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
        ) as progress:
            task = progress.add_task(
                f"Embedding (batch={current_batch_size})...",
                total=total_chunks
            )

            while position < total_chunks:
                batch_end = min(position + current_batch_size, total_chunks)
                batch_texts = [c.content for c in chunks[position:batch_end]]

                try:
                    batch_vectors = _embed_batch_native(embedder, batch_texts)
                    vectors.extend(batch_vectors)
                    position = batch_end
                    progress.update(task, advance=len(batch_texts))

                except Exception as e:
                    if _is_batch_size_error(e) and current_batch_size > 1:
                        # Halve batch size and retry
                        current_batch_size = max(1, current_batch_size // 2)
                        progress.update(
                            task,
                            description=f"Embedding (batch={current_batch_size})..."
                        )
                        if current_batch_size == 1:
                            _print(f"Falling back to single embedding", "yellow" if RICH else "")
                    else:
                        # Not a batch size error or already at batch_size=1
                        raise
    else:
        # Non-rich progress
        while position < total_chunks:
            batch_end = min(position + current_batch_size, total_chunks)
            batch_texts = [c.content for c in chunks[position:batch_end]]

            try:
                batch_vectors = _embed_batch_native(embedder, batch_texts)
                vectors.extend(batch_vectors)

                if show_progress:
                    print(f"  Embedded {batch_end}/{total_chunks} (batch={current_batch_size})...")

                position = batch_end

            except Exception as e:
                if _is_batch_size_error(e) and current_batch_size > 1:
                    # Halve batch size and retry
                    old_size = current_batch_size
                    current_batch_size = max(1, current_batch_size // 2)
                    if show_progress:
                        print(f"  Batch size {old_size} failed, trying {current_batch_size}...")
                else:
                    raise

    return vectors


def _embed_chunks_sequential(
        embedder: Any,
        chunks: List[Chunk],
        show_progress: bool = True,
) -> List[List[float]]:
    """Embed chunks one at a time (fallback when batch not available)."""
    vectors: List[List[float]] = []
    total_chunks = len(chunks)

    if RICH and show_progress and total_chunks > 1:
        with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
        ) as progress:
            task = progress.add_task("Embedding...", total=total_chunks)
            for chunk in chunks:
                vectors.append(embedder.embed(chunk.content))
                progress.update(task, advance=1)
    else:
        for i, chunk in enumerate(chunks):
            vectors.append(embedder.embed(chunk.content))
            if show_progress and (i + 1) % 20 == 0:
                print(f"  Embedded {i + 1}/{total_chunks}...")

    return vectors


# =============================================================================
# Main Command
# =============================================================================


def command(
        source: Optional[Path] = typer.Argument(
            None,
            help="Source file or directory (will prompt if not provided).",
        ),
        non_interactive: bool = typer.Option(
            False,
            "--non-interactive",
            "-y",
            help="Use defaults without prompting.",
        ),
) -> None:
    """
    Ingest documents into the vector database.

    Run without arguments for interactive mode:
        fitz ingest

    Or provide source directly:
        fitz ingest ./docs

    Use -y for non-interactive mode with defaults:
        fitz ingest ./docs -y
    """
    # =========================================================================
    # Load config (for defaults)
    # =========================================================================

    config = _load_config()

    # Extract defaults from config
    embedding_plugin = config.get("embedding", {}).get("plugin_name", "cohere")
    vector_db_plugin = config.get("vector_db", {}).get("plugin_name", "qdrant")
    default_collection = config.get("retriever", {}).get("collection", "default")

    # Get available chunkers
    available_chunkers = available_chunking_plugins()
    if not available_chunkers:
        available_chunkers = ["simple"]

    # =========================================================================
    # Header
    # =========================================================================

    _header("Fitz Ingest")

    if RICH:
        console.print(f"[dim]Embedding:[/dim] {embedding_plugin}")
        console.print(f"[dim]Vector DB:[/dim] {vector_db_plugin}")
        console.print()
    else:
        print(f"Embedding: {embedding_plugin}")
        print(f"Vector DB: {vector_db_plugin}")
        print()

    # =========================================================================
    # Interactive Prompts (or use defaults)
    # =========================================================================

    if non_interactive:
        # Use defaults
        if source is None:
            _error("Source path required in non-interactive mode.")
            _print("Usage: fitz ingest ./docs -y", "dim")
            raise typer.Exit(1)

        collection = default_collection
        chunker = "simple"
        chunk_size = 1000
        chunk_overlap = 0

        _print(f"Source: {source}", "dim")
        _print(f"Collection: {collection}", "dim")
        _print(f"Chunker: {chunker} (size={chunk_size})", "dim")

    else:
        # Interactive mode
        _print("[bold]Configure ingestion:[/bold]" if RICH else "Configure ingestion:")
        print()

        # Source path
        if source is None:
            source = _prompt_path("Source path (file or directory)", ".")
        else:
            # Validate provided source
            if not source.exists():
                _error(f"Source does not exist: {source}")
                raise typer.Exit(1)
            _print(f"Source: {source}", "dim")

        # Collection name
        collection = _prompt_text("Collection name", default_collection)

        # Chunking strategy
        chunker = _prompt_choice(
            "Chunking strategy",
            choices=available_chunkers,
            default="simple",
        )

        # Chunk size
        chunk_size = _prompt_int("Chunk size (characters)", 1000)

        # Chunk overlap
        chunk_overlap = _prompt_int("Chunk overlap", 0)

        print()

    # =========================================================================
    # Confirm before proceeding
    # =========================================================================

    if not non_interactive:
        if RICH:
            console.print(Panel(
                f"[bold]Source:[/bold] {source}\n"
                f"[bold]Collection:[/bold] {collection}\n"
                f"[bold]Chunker:[/bold] {chunker} (size={chunk_size}, overlap={chunk_overlap})\n"
                f"[bold]Embedding:[/bold] {embedding_plugin}\n"
                f"[bold]Vector DB:[/bold] {vector_db_plugin}",
                title="Summary",
                border_style="green",
            ))

            if not Confirm.ask("Proceed with ingestion?", default=True):
                _print("Cancelled.", "yellow")
                raise typer.Exit(0)
        else:
            print("\nSummary:")
            print(f"  Source: {source}")
            print(f"  Collection: {collection}")
            print(f"  Chunker: {chunker} (size={chunk_size}, overlap={chunk_overlap})")
            print(f"  Embedding: {embedding_plugin}")
            print(f"  Vector DB: {vector_db_plugin}")

            confirm = input("\nProceed? [Y/n]: ").strip().lower()
            if confirm in ("n", "no"):
                print("Cancelled.")
                raise typer.Exit(0)

        print()

    # =========================================================================
    # Step 1: Read documents
    # =========================================================================

    _step(1, 4, "Reading documents...")

    try:
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin, kwargs={})
        raw_docs = list(ingest_engine.run(str(source)))
    except Exception as e:
        _error(f"Failed to read documents: {e}")
        raise typer.Exit(1)

    if not raw_docs:
        _error("No documents found.")
        raise typer.Exit(1)

    _success(f"Found {len(raw_docs)} documents")

    # =========================================================================
    # Step 2: Chunk documents
    # =========================================================================

    _step(2, 4, f"Chunking ({chunker})...")

    chunker_kwargs = {"chunk_size": chunk_size}
    if chunk_overlap > 0:
        chunker_kwargs["chunk_overlap"] = chunk_overlap

    chunker_config = ChunkerConfig(plugin_name=chunker, kwargs=chunker_kwargs)

    try:
        chunking_engine = ChunkingEngine.from_config(chunker_config)
    except Exception as e:
        _error(f"Failed to initialize chunker: {e}")
        raise typer.Exit(1)

    chunks: List[Chunk] = []
    for raw_doc in raw_docs:
        try:
            doc_chunks = chunking_engine.run(raw_doc)
            chunks.extend(doc_chunks)
        except Exception as e:
            logger.warning(f"Failed to chunk {getattr(raw_doc, 'path', '?')}: {e}")

    if not chunks:
        _error("No chunks created.")
        raise typer.Exit(1)

    _success(f"Created {len(chunks)} chunks")

    # =========================================================================
    # Step 3: Generate embeddings
    # =========================================================================

    _step(3, 4, f"Embedding ({embedding_plugin})...")

    try:
        embedder = get_llm_plugin(plugin_type="embedding", plugin_name=embedding_plugin)
    except Exception as e:
        _error(f"Failed to initialize embedder: {e}")
        raise typer.Exit(1)

    # Use batch size of 96 (Cohere max) for efficiency
    embed_batch_size = 96

    try:
        vectors = _embed_chunks(embedder, chunks, batch_size=embed_batch_size)
    except Exception as e:
        _error(f"Embedding failed: {e}")
        raise typer.Exit(1)

    _success(f"Generated {len(vectors)} embeddings")

    # =========================================================================
    # Step 4: Write to vector DB
    # =========================================================================

    _step(4, 4, f"Writing to {vector_db_plugin}...")

    try:
        vdb = get_vector_db_plugin(vector_db_plugin)
        writer = VectorDBWriter(client=vdb)
    except Exception as e:
        _error(f"Failed to connect to vector DB: {e}")
        raise typer.Exit(1)

    batch_size = 50
    try:
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            batch_vectors = vectors[i:i + batch_size]
            writer.upsert(collection=collection, chunks=batch_chunks, vectors=batch_vectors)
    except Exception as e:
        _error(f"Failed to write: {e}")
        raise typer.Exit(1)

    _success(f"Written to '{collection}'")

    # =========================================================================
    # Done
    # =========================================================================

    print()
    if RICH:
        console.print(Panel.fit(
            f"[green bold]Success![/green bold]\n\n"
            f"Ingested [bold]{len(chunks)}[/bold] chunks into [bold]'{collection}'[/bold]\n\n"
            f"[dim]Query with:[/dim] fitz query \"your question\"",
            border_style="green",
        ))
    else:
        print(f"Success! Ingested {len(chunks)} chunks into '{collection}'")
        print(f"\nQuery with: fitz query \"your question\"")
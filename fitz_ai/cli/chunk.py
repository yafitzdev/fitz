# fitz/cli/chunk.py
"""
Chunk command: Preview how documents will be chunked.

Usage:
    fitz chunk ./doc.pdf                     # Preview chunking with defaults
    fitz chunk ./doc.pdf --size 500          # Custom chunk size
    fitz chunk ./doc.pdf --chunker simple    # Specific chunker
    fitz chunk ./docs/ --stats               # Just show stats
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

# Try to import rich for pretty output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.syntax import Syntax
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


def _read_file_content(path: Path) -> str:
    """Read file content, handling different file types."""
    # For now, just read as text
    # TODO: Add PDF extraction, etc.
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        raise typer.BadParameter(f"Could not read file: {e}")


def _get_content_from_source(source: Path) -> List[tuple[Path, str]]:
    """Get content from file or directory."""
    results = []

    if source.is_file():
        content = _read_file_content(source)
        results.append((source, content))
    elif source.is_dir():
        # Get all text-ish files
        extensions = {".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".html", ".xml"}
        for file_path in source.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                try:
                    content = _read_file_content(file_path)
                    results.append((file_path, content))
                except Exception:
                    pass  # Skip unreadable files

    return results


def command(
    source: Path = typer.Argument(
        ...,
        help="File or directory to chunk.",
    ),
    chunker: str = typer.Option(
        "simple",
        "--chunker",
        "-c",
        help="Chunking plugin to use.",
    ),
    size: int = typer.Option(
        1000,
        "--size",
        "-s",
        help="Target chunk size in characters.",
    ),
    overlap: int = typer.Option(
        0,
        "--overlap",
        "-o",
        help="Overlap between chunks in characters.",
    ),
    stats_only: bool = typer.Option(
        False,
        "--stats",
        help="Only show statistics, no chunk content.",
    ),
    show_all: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all chunks (default: first 5).",
    ),
    limit: int = typer.Option(
        5,
        "--limit",
        "-n",
        help="Number of chunks to preview.",
    ),
    list_chunkers: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List available chunking plugins.",
    ),
) -> None:
    """
    Preview how documents will be chunked.

    Use this to tune chunking parameters BEFORE running expensive embeddings.

    Examples:
        fitz chunk ./doc.txt                  # Preview with defaults
        fitz chunk ./doc.txt --size 500       # Smaller chunks
        fitz chunk ./docs/ --stats            # Stats for whole directory
        fitz chunk --list                     # Show available chunkers
    """
    from fitz_ai.core.registry import available_chunking_plugins, get_chunking_plugin

    # List mode
    if list_chunkers:
        chunkers = available_chunking_plugins()
        typer.echo()
        typer.echo("Available chunking plugins:")
        for name in chunkers:
            typer.echo(f"  • {name}")
        typer.echo()
        typer.echo("Usage: fitz chunk ./file.txt --chunker <name>")
        return

    # Validate source
    if not source.exists():
        typer.echo(f"Error: {source} does not exist")
        raise typer.Exit(1)

    # Get chunker
    try:
        ChunkerCls = get_chunking_plugin(chunker)
    except Exception:
        typer.echo(f"Error: Unknown chunker '{chunker}'")
        typer.echo()
        typer.echo("Available chunkers:")
        for name in available_chunking_plugins():
            typer.echo(f"  • {name}")
        raise typer.Exit(1)

    # Initialize chunker with options
    chunker_kwargs = {"chunk_size": size}
    if overlap > 0:
        chunker_kwargs["chunk_overlap"] = overlap

    try:
        chunker_instance = ChunkerCls(**chunker_kwargs)
    except TypeError:
        # Some chunkers may not support all kwargs
        chunker_instance = ChunkerCls(chunk_size=size)

    # Get content
    files_content = _get_content_from_source(source)

    if not files_content:
        typer.echo("No readable files found.")
        raise typer.Exit(1)

    # Chunk all files
    all_chunks = []
    file_stats = []

    for file_path, content in files_content:
        base_meta = {
            "source_file": str(file_path),
            "doc_id": str(file_path),
        }
        chunks = chunker_instance.chunk_text(content, base_meta)
        all_chunks.extend(chunks)

        # Calculate stats for this file
        chunk_sizes = [len(c.content) for c in chunks]
        file_stats.append(
            {
                "file": file_path,
                "content_chars": len(content),
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) // len(chunk_sizes) if chunk_sizes else 0,
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
            }
        )

    # Calculate totals
    total_chars = sum(s["content_chars"] for s in file_stats)
    total_chunks = len(all_chunks)
    all_sizes = [len(c.content) for c in all_chunks]
    avg_size = sum(all_sizes) // len(all_sizes) if all_sizes else 0

    # Display header
    if RICH_AVAILABLE:
        console.print(
            Panel.fit(
                f"[bold]Chunking Preview[/bold]\n"
                f"[dim]Chunker: {chunker} | Size: {size} | Overlap: {overlap}[/dim]",
                title="✂️  fitz chunk",
                border_style="blue",
            )
        )
    else:
        typer.echo()
        typer.echo("=" * 60)
        typer.echo("Chunking Preview")
        typer.echo(f"Chunker: {chunker} | Size: {size} | Overlap: {overlap}")
        typer.echo("=" * 60)

    # Display stats
    typer.echo()
    if RICH_AVAILABLE:
        # Summary table
        table = Table(title="Summary", show_header=False, box=None)
        table.add_column("Property", style="dim")
        table.add_column("Value", style="bold")

        table.add_row("Files", str(len(files_content)))
        table.add_row("Total characters", f"{total_chars:,}")
        table.add_row("Total chunks", str(total_chunks))
        table.add_row("Avg chunk size", f"{avg_size:,} chars")
        table.add_row("Min chunk size", f"{min(all_sizes):,} chars" if all_sizes else "N/A")
        table.add_row("Max chunk size", f"{max(all_sizes):,} chars" if all_sizes else "N/A")

        console.print(table)

        # Per-file breakdown if multiple files
        if len(file_stats) > 1:
            console.print()
            file_table = Table(title="Per-File Breakdown")
            file_table.add_column("File", style="cyan")
            file_table.add_column("Chars", justify="right")
            file_table.add_column("Chunks", justify="right")
            file_table.add_column("Avg Size", justify="right")

            for stat in file_stats:
                file_table.add_row(
                    stat["file"].name,
                    f"{stat['content_chars']:,}",
                    str(stat["num_chunks"]),
                    f"{stat['avg_chunk_size']:,}",
                )

            console.print(file_table)
    else:
        typer.echo(f"Files:           {len(files_content)}")
        typer.echo(f"Total chars:     {total_chars:,}")
        typer.echo(f"Total chunks:    {total_chunks}")
        typer.echo(f"Avg chunk size:  {avg_size:,} chars")
        typer.echo(f"Min chunk size:  {min(all_sizes):,} chars" if all_sizes else "N/A")
        typer.echo(f"Max chunk size:  {max(all_sizes):,} chars" if all_sizes else "N/A")

    # If stats only, stop here
    if stats_only:
        typer.echo()
        return

    # Show chunk previews
    typer.echo()

    num_to_show = len(all_chunks) if show_all else min(limit, len(all_chunks))

    if RICH_AVAILABLE:
        console.print(f"[bold]Chunk Previews[/bold] (showing {num_to_show} of {total_chunks}):")
    else:
        typer.echo(f"Chunk Previews (showing {num_to_show} of {total_chunks}):")
        typer.echo("-" * 60)

    for i, chunk in enumerate(all_chunks[:num_to_show]):
        content_preview = chunk.content
        if len(content_preview) > 300:
            content_preview = content_preview[:300] + "..."

        # Clean up for display
        content_preview = content_preview.replace("\n", " ").strip()

        if RICH_AVAILABLE:
            console.print()
            console.print(
                f"[dim]#{i + 1}[/dim] "
                f"[cyan]{Path(chunk.doc_id).name}[/cyan] "
                f"[dim]({len(chunk.content):,} chars)[/dim]"
            )
            console.print(f"  [dim]{content_preview}[/dim]")
        else:
            typer.echo()
            typer.echo(f"#{i + 1} {Path(chunk.doc_id).name} ({len(chunk.content):,} chars)")
            typer.echo(f"  {content_preview}")

    # Helpful footer
    typer.echo()
    if total_chunks > num_to_show:
        if RICH_AVAILABLE:
            console.print(
                f"[dim]Use --all to see all {total_chunks} chunks, or --limit N to see more[/dim]"
            )
        else:
            typer.echo(f"Use --all to see all {total_chunks} chunks, or --limit N to see more")

    typer.echo()
    if RICH_AVAILABLE:
        console.print("[dim]Happy with the chunking? Run:[/dim]")
        console.print(f"  fitz ingest {source} <collection> --chunk-size {size}")
    else:
        typer.echo("Happy with the chunking? Run:")
        typer.echo(f"  fitz ingest {source} <collection> --chunk-size {size}")
    typer.echo()

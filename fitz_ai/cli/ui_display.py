# fitz_ai/cli/ui_display.py
"""
Shared display functions for CLI commands.

This module provides consistent output formatting across commands
like `fitz query` and `fitz quickstart`.
"""

from __future__ import annotations

import os

from fitz_ai.cli.ui import RICH, Markdown, Panel, Table, console


def display_answer(answer, show_sources: bool = True) -> None:
    """
    Display an RGS answer with optional sources.

    Used by both `fitz query` and `fitz quickstart` for consistent output.

    Args:
        answer: RGSAnswer object with .answer and .sources attributes
        show_sources: Whether to show source documents
    """
    print()

    if RICH:
        # Answer panel
        console.print(
            Panel(
                Markdown(answer.answer),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            )
        )

        # Sources table
        if show_sources and hasattr(answer, "sources") and answer.sources:
            print()
            table = Table(title="Sources")
            table.add_column("#", style="dim", width=3)
            table.add_column("Document", style="cyan", max_width=40)
            table.add_column("Chunk", style="dim", justify="center", width=5)
            table.add_column("Vector", style="yellow", justify="right", width=7)
            table.add_column("Rerank", style="green", justify="right", width=7)
            table.add_column("Excerpt", style="dim", max_width=45)

            for i, source in enumerate(answer.sources[:5], 1):
                doc_id = getattr(source, "doc_id", getattr(source, "source_file", "?"))
                content = getattr(source, "content", getattr(source, "text", ""))
                metadata = getattr(source, "metadata", {})

                # Get filename only (not full path)
                filename = os.path.basename(doc_id) if doc_id else "?"

                # Get title if available, otherwise use filename
                title = metadata.get("title", "")
                display_name = title if title else filename

                # Truncate display name if too long
                if len(display_name) > 38:
                    display_name = display_name[:35] + "..."

                # Get chunk index
                chunk_idx = metadata.get("chunk_index", "-")
                chunk_str = str(chunk_idx) if chunk_idx != "-" else "-"

                # Get scores
                vector_score = metadata.get("vector_score")
                rerank_score = metadata.get("rerank_score")
                vector_str = f"{vector_score:.3f}" if vector_score is not None else "-"
                rerank_str = f"{rerank_score:.3f}" if rerank_score is not None else "-"

                # Excerpt
                excerpt = content[:70] + "..." if len(content) > 70 else content
                excerpt = excerpt.replace("\n", " ").replace("\r", " ")

                table.add_row(
                    str(i), display_name, chunk_str, vector_str, rerank_str, excerpt
                )

            console.print(table)
    else:
        # Plain text output
        print("Answer:")
        print("-" * 40)
        print(answer.answer)
        print()

        if show_sources and hasattr(answer, "sources") and answer.sources:
            print("Sources:")
            for i, source in enumerate(answer.sources[:5], 1):
                doc_id = getattr(source, "doc_id", getattr(source, "source_file", "?"))
                metadata = getattr(source, "metadata", {})

                # Get filename only
                filename = os.path.basename(doc_id) if doc_id else "?"
                title = metadata.get("title", "")
                display_name = title if title else filename

                # Get chunk index
                chunk_idx = metadata.get("chunk_index", "")
                chunk_str = f" [chunk {chunk_idx}]" if chunk_idx != "" else ""

                # Get scores
                vector_score = metadata.get("vector_score")
                rerank_score = metadata.get("rerank_score")
                scores = []
                if vector_score is not None:
                    scores.append(f"vec={vector_score:.3f}")
                if rerank_score is not None:
                    scores.append(f"rerank={rerank_score:.3f}")
                score_str = f" ({', '.join(scores)})" if scores else ""

                print(f"  [{i}] {display_name}{chunk_str}{score_str}")


def display_sources(chunks, max_sources: int = 5, indent: int = 0) -> None:
    """
    Display source chunks in a table.

    Used by `fitz query` and `fitz chat` for consistent output.

    Args:
        chunks: List of Chunk objects with content and metadata
        max_sources: Maximum number of sources to display
        indent: Left padding/indent in spaces
    """
    if not chunks:
        return

    print()

    if RICH:
        from rich.padding import Padding

        table = Table(title="Sources")
        table.add_column("#", style="dim", width=3)
        table.add_column("Document", style="cyan", max_width=40)
        table.add_column("Chunk", style="dim", justify="center", width=5)
        table.add_column("Vector", style="yellow", justify="right", width=7)
        table.add_column("Rerank", style="green", justify="right", width=7)
        table.add_column("Excerpt", style="dim", max_width=45)

        for i, chunk in enumerate(chunks[:max_sources], 1):
            # Get doc_id from chunk
            doc_id = getattr(chunk, "doc_id", None)
            if not doc_id:
                doc_id = getattr(chunk, "metadata", {}).get("source_file", "?")

            # Get filename only (not full path)
            filename = os.path.basename(doc_id) if doc_id else "?"

            # Truncate display name if too long
            if len(filename) > 38:
                filename = filename[:35] + "..."

            # Get metadata
            metadata = getattr(chunk, "metadata", {})

            # Get chunk index
            chunk_idx = metadata.get("chunk_index", "-")
            chunk_str = str(chunk_idx) if chunk_idx != "-" else "-"

            # Get scores
            vector_score = metadata.get("vector_score")
            rerank_score = metadata.get("rerank_score")
            vector_str = f"{vector_score:.3f}" if vector_score is not None else "-"
            rerank_str = f"{rerank_score:.3f}" if rerank_score is not None else "-"

            # Excerpt
            content = getattr(chunk, "content", str(chunk))
            excerpt = content[:70] + "..." if len(content) > 70 else content
            excerpt = excerpt.replace("\n", " ").replace("\r", " ")

            table.add_row(str(i), filename, chunk_str, vector_str, rerank_str, excerpt)

        if indent > 0:
            console.print(Padding(table, (0, 0, 0, indent)))
        else:
            console.print(table)
    else:
        print("Sources:")
        for i, chunk in enumerate(chunks[:max_sources], 1):
            doc_id = getattr(chunk, "doc_id", None)
            if not doc_id:
                doc_id = getattr(chunk, "metadata", {}).get("source_file", "?")
            filename = os.path.basename(doc_id) if doc_id else "?"

            metadata = getattr(chunk, "metadata", {})
            chunk_idx = metadata.get("chunk_index", "")
            chunk_str = f" [chunk {chunk_idx}]" if chunk_idx != "" else ""

            print(f"  [{i}] {filename}{chunk_str}")


__all__ = ["display_answer", "display_sources"]

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


__all__ = ["display_answer"]

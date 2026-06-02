# fitz_sage/cli/ui/display.py
"""
Display functions for answers and sources.

Provides consistent output formatting for query results.
"""

from __future__ import annotations

import os
import sys

from .console import RICH, Markdown, Panel, Table, console


def _sanitize_for_display(text: str) -> str:
    """Sanitize text for Windows terminal display (replace problematic Unicode)."""
    if sys.platform == "win32":
        # Replace common Unicode arrows with ASCII
        text = text.replace("→", "->").replace("←", "<-")
        text = text.replace("⟶", "-->").replace("⟵", "<--")
    return text


def display_answer(answer, show_sources: bool = True) -> None:
    """
    Display an answer with optional sources.

    Used by `fitz query` for consistent output.
    Supports both core Answer (.text, .provenance) and RGSAnswer (.answer, .sources).

    Args:
        answer: Answer object (core or RGS format)
        show_sources: Whether to show source documents
    """
    print()

    # Support both Answer.text and RGSAnswer.answer
    answer_text = getattr(answer, "text", None) or getattr(answer, "answer", "")

    # Support both Answer.provenance and RGSAnswer.sources
    sources = getattr(answer, "provenance", None) or getattr(answer, "sources", [])

    if RICH:
        # Answer panel
        console.print(
            Panel(
                Markdown(answer_text),
                title="[bold green]Answer[/bold green]",
                border_style="green",
            )
        )

        # Sources table
        if show_sources and sources:
            print()
            table = Table(title="Sources")
            table.add_column("#", style="dim", width=3)
            table.add_column("File", style="cyan", max_width=50)
            table.add_column("Location", style="yellow", max_width=20)
            table.add_column("Excerpt", style="dim", max_width=45)

            for i, source in enumerate(sources[:5], 1):
                metadata = getattr(source, "metadata", {})
                content = getattr(source, "excerpt", None) or getattr(
                    source, "content", getattr(source, "text", "")
                )

                # Resolve display name: file_path > disk_path > source_id
                file_path = metadata.get("file_path") or metadata.get("disk_path", "")
                if file_path:
                    parts = file_path.replace("\\", "/").split("/")
                    display_name = "/".join(parts[-2:]) if len(parts) > 1 else parts[0]
                else:
                    doc_id = getattr(source, "source_id", None) or getattr(
                        source, "doc_id", getattr(source, "source_file", "?")
                    )
                    display_name = os.path.basename(doc_id) if doc_id else "?"
                    title = metadata.get("title", "")
                    if title:
                        display_name = title

                # Location: line range or section title
                location = ""
                line_range = metadata.get("line_range")
                if line_range:
                    location = f"L{line_range[0]}-{line_range[1]}"
                elif metadata.get("section_title"):
                    location = metadata["section_title"][:18]

                # Excerpt
                excerpt = content[:70] + "..." if len(content) > 70 else content
                excerpt = excerpt.replace("\n", " ").replace("\r", " ")
                excerpt = _sanitize_for_display(excerpt)

                table.add_row(str(i), display_name, location, excerpt)

            console.print(table)
    else:
        # Plain text output
        print("Answer:")
        print("-" * 40)
        print(answer_text)
        print()

        if show_sources and sources:
            print("Sources:")
            for i, source in enumerate(sources[:5], 1):
                metadata = getattr(source, "metadata", {})

                file_path = metadata.get("file_path") or metadata.get("disk_path", "")
                if file_path:
                    display_name = file_path.replace("\\", "/")
                else:
                    doc_id = getattr(source, "source_id", None) or getattr(
                        source, "doc_id", getattr(source, "source_file", "?")
                    )
                    display_name = os.path.basename(doc_id) if doc_id else "?"

                line_range = metadata.get("line_range")
                loc = f" L{line_range[0]}-{line_range[1]}" if line_range else ""

                print(f"  [{i}] {display_name}{loc}")


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

            # Get score
            rerank_score = metadata.get("rerank_score")
            rerank_str = f"{rerank_score:.3f}" if rerank_score is not None else "-"

            # Excerpt
            content = getattr(chunk, "content", str(chunk))
            excerpt = content[:70] + "..." if len(content) > 70 else content
            excerpt = excerpt.replace("\n", " ").replace("\r", " ")
            excerpt = _sanitize_for_display(excerpt)

            table.add_row(str(i), filename, chunk_str, rerank_str, excerpt)

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


def display_evidence_pack(pack, max_items: int = 10) -> None:
    """Display a governed evidence pack."""
    print()

    mode = getattr(pack, "mode", None)
    mode_text = getattr(mode, "value", mode) or "unlabeled"
    reasons = getattr(pack, "reasons", []) or []
    items = getattr(pack, "items", []) or []
    indexing_status = getattr(pack, "indexing_status", {}) or {}
    metadata = getattr(pack, "metadata", {}) or {}
    governance_lines = _format_governance_metadata(metadata, reasons)

    if RICH:
        table = Table(title=_evidence_title(mode_text, metadata))
        table.add_column("#", style="dim", width=3)
        table.add_column("File", style="cyan", max_width=48)
        table.add_column("Location", style="yellow", max_width=24)
        table.add_column("Score", style="magenta", width=8)
        table.add_column("Excerpt", style="dim", max_width=64)

        for item in items[:max_items]:
            score = "-" if item.score is None else f"{item.score:.3f}"
            location = item.address_location
            if item.line_range:
                location = f"{location}:{item.line_range[0]}-{item.line_range[1]}"
            table.add_row(
                str(item.rank),
                _short_path(item.file_path),
                location,
                score,
                _compact_evidence_excerpt(item.excerpt),
            )
        if governance_lines:
            table.caption = "\n".join(governance_lines)
        console.print(table)
    else:
        print(_evidence_title(mode_text, metadata))
        for line in governance_lines:
            print(line)
        for item in items[:max_items]:
            score = "-" if item.score is None else f"{item.score:.3f}"
            location = item.address_location
            if item.line_range:
                location = f"{location}:{item.line_range[0]}-{item.line_range[1]}"
            print(f"[{item.rank}] {_short_path(item.file_path)} {location} score={score}")
            print(f"    {_compact_evidence_excerpt(item.excerpt)}")

    if indexing_status and (
        not indexing_status.get("complete", True) or not indexing_status.get("fully_enriched", True)
    ):
        status_line = _format_indexing_status(indexing_status)
        if RICH:
            console.print(f"[dim]{status_line}[/dim]")
        else:
            print(status_line)


def _format_indexing_status(indexing_status: dict) -> str:
    """Return a user-facing status line for query-ready vs deep enrichment work."""
    total = indexing_status.get("total", "?")
    by_state = indexing_status.get("by_state", {}) or {}
    if by_state and not by_state.get("registered", 0):
        if not indexing_status.get("complete", True):
            pending = indexing_status.get("pending", "?")
            return f"Enrichment pending: {pending}/{total}"
        if not indexing_status.get("fully_enriched", True):
            pending = indexing_status.get("deep_pending", "?")
            return f"Deep enrichment pending: {pending}/{total}"

    if indexing_status.get("query_ready") and not indexing_status.get("fully_enriched", True):
        pending = indexing_status.get("deep_pending", "?")
        return f"Deep enrichment pending: {pending}/{total}"

    pending = indexing_status.get("pending", "?")
    return f"Indexing pending: {pending}/{total}"


def _evidence_title(mode_text: str, metadata: dict) -> str:
    """Return the evidence table title with Pyrrho verdict folded in."""
    if _is_broad_overview(metadata):
        return "Representative Sources (Broad Overview)"
    pyrrho = _pyrrho_metadata(metadata)
    verdict = pyrrho.get("mode") if pyrrho else mode_text
    return f"Evidence (Pyrrho: {_format_verdict(verdict)})" if verdict else "Evidence"


def _format_governance_metadata(metadata: dict, reasons: list[str]) -> list[str]:
    """Format Pyrrho and cutoff metadata as compact display lines."""
    cutoff = metadata.get("governance_cutoff", {}) if isinstance(metadata, dict) else {}
    if not isinstance(cutoff, dict):
        cutoff = {}

    lines: list[str] = []
    if _is_broad_overview(metadata):
        selected = cutoff.get("selected", "?")
        max_items = cutoff.get("max", "?")
        lines.append(
            f"Broad overview: selected {selected} representative source(s) from top {max_items}; "
            "evidence sufficiency was not evaluated."
        )
        return _append_unique_reasons(lines, reasons)

    pyrrho = _pyrrho_metadata(metadata)
    probs = pyrrho.get("probabilities", {}) if pyrrho else {}
    if isinstance(probs, dict) and probs:
        lines.append(
            "Pyrrho: "
            f"P(TRUSTWORTHY)={_fmt_prob(probs.get('trustworthy'))}  "
            f"P(ABSTAIN)={_fmt_prob(probs.get('abstain'))}  "
            f"P(DISPUTED)={_fmt_prob(probs.get('disputed'))}"
        )

    policy = cutoff.get("policy", {}) if isinstance(cutoff.get("policy", {}), dict) else {}
    has_cutoff_values = any(key in cutoff for key in ("selected", "evaluated", "max")) or bool(
        policy
    )
    if has_cutoff_values:
        parts = [
            f"selected {cutoff.get('selected', '?')}",
            f"evaluated {cutoff.get('evaluated', '?')}/{cutoff.get('max', '?')}",
        ]
        if policy:
            parts.extend(
                [
                    f"policy {policy.get('query_shape', '?')}",
                    f"min trustworthy {policy.get('min_trustworthy_docs', '?')}",
                    f"min disputed {policy.get('min_disputed_docs', '?')}",
                    f"dispute patience {policy.get('disputed_patience_docs', '?')}",
                ]
            )
        lines.append("Cutoff: " + "; ".join(parts))

    reason = pyrrho.get("reason") if pyrrho else None
    shown_reasons: set[str] = set()
    if isinstance(reason, str) and reason:
        lines.append(reason)
        shown_reasons.add(reason)
    return _append_unique_reasons(lines, reasons, shown_reasons=shown_reasons)


def _append_unique_reasons(
    lines: list[str],
    reasons: list[str],
    *,
    shown_reasons: set[str] | None = None,
) -> list[str]:
    """Append reasons once while preserving order."""
    shown = shown_reasons or set(lines)
    for item in reasons:
        if item not in shown:
            lines.append(item)
            shown.add(item)
    return lines


def _is_broad_overview(metadata: dict) -> bool:
    """Return whether metadata represents a deterministic broad-overview result."""
    cutoff = metadata.get("governance_cutoff", {}) if isinstance(metadata, dict) else {}
    if not isinstance(cutoff, dict):
        return False
    policy = cutoff.get("policy", {}) if isinstance(cutoff.get("policy", {}), dict) else {}
    return (
        bool(cutoff.get("representative_sources")) or policy.get("query_shape") == "broad_overview"
    )


def _pyrrho_metadata(metadata: dict) -> dict:
    """Return nested Pyrrho metadata from an evidence pack metadata dict."""
    cutoff = metadata.get("governance_cutoff", {}) if isinstance(metadata, dict) else {}
    if not isinstance(cutoff, dict):
        return {}
    pyrrho = cutoff.get("pyrrho", {})
    return pyrrho if isinstance(pyrrho, dict) else {}


def _fmt_prob(value: object) -> str:
    """Format a probability-like value."""
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "?"


def _format_verdict(value: object) -> str:
    """Format a governance verdict for a compact table title."""
    text = str(value).strip()
    return text.upper() if text else "UNKNOWN"


def _compact_evidence_excerpt(text: str, max_chars: int = 96) -> str:
    """Return a terminal-sized excerpt without changing the JSON payload."""
    compact = " ".join(str(text).split())
    if len(compact) > max_chars:
        compact = compact[: max_chars - 3].rstrip() + "..."
    return _sanitize_for_display(compact)


def _short_path(path: str) -> str:
    """Return a compact path for terminal display."""
    if not path:
        return "?"
    parts = path.replace("\\", "/").split("/")
    return "/".join(parts[-3:]) if len(parts) > 3 else path


__all__ = ["display_answer", "display_sources", "display_evidence_pack"]

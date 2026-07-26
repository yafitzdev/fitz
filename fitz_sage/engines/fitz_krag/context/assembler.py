# fitz_sage/engines/fitz_krag/context/assembler.py
"""
Context assembler — formats read results into LLM-ready context blocks.

Output format:
    [S1] # path/to/file.py (lines 10-25)
    ```python
    def function_name(...):
        ...
    ```

    [S2] # path/to/other.py (lines 5-15)
    ```python
    class ClassName:
        ...
    ```
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from fitz_sage.engines.fitz_krag.types import AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

logger = logging.getLogger(__name__)

# Rough token estimation: ~4 chars per token
CHARS_PER_TOKEN = 4


class ContextAssembler:
    """Assembles read results into LLM-ready context blocks."""

    def __init__(self, config: "FitzKragConfig"):
        self._config = config

    def assemble(self, query: str, results: list[ReadResult]) -> str:
        """
        Format read results as numbered source blocks.

        When results contain mixed types (code + documentation), groups them
        with type headers. Adds [S1], [S2] markers for citation.
        Respects max_context_tokens budget.
        """
        if not results:
            return ""

        # Check if results are mixed-type (code + documentation)
        has_mixed = self._is_mixed_type(results)

        blocks: list[str] = []
        budget = self._config.max_context_tokens * CHARS_PER_TOKEN
        used = 0
        current_group: str | None = None

        for i, result in enumerate(results):
            # Add type group header for mixed-type results
            if has_mixed:
                group = self._type_group(result)
                if group != current_group:
                    header = f"--- {group} ---"
                    if used + len(header) + 2 < budget:
                        blocks.append(header)
                        used += len(header) + 2
                    current_group = group

            block = self._format_block(i + 1, result)
            block_len = len(block)

            if used + block_len > budget:
                remaining = budget - used
                if remaining > 100:
                    block = block[:remaining] + "\n```\n(truncated)"
                    blocks.append(block)
                break

            blocks.append(block)
            used += block_len

        return "\n\n".join(blocks)

    def _is_mixed_type(self, results: list[ReadResult]) -> bool:
        """Check if results contain multiple knowledge types."""
        kinds = {r.address.kind for r in results}
        code_kinds = {AddressKind.SYMBOL, AddressKind.FILE}
        doc_kinds = {AddressKind.SECTION}
        data_kinds = {AddressKind.TABLE}
        type_groups = sum(
            [
                bool(kinds & code_kinds),
                bool(kinds & doc_kinds),
                bool(kinds & data_kinds),
            ]
        )
        return type_groups >= 2

    def _type_group(self, result: ReadResult) -> str:
        """Get the type group label for a result."""
        if result.address.kind in (AddressKind.SYMBOL, AddressKind.FILE):
            return "Code"
        elif result.address.kind == AddressKind.SECTION:
            return "Documentation"
        elif result.address.kind == AddressKind.TABLE:
            return "Data"
        return "Other"

    def _format_block(self, index: int, result: ReadResult) -> str:
        """Format a single source block."""
        header = self._format_header(index, result)
        lang = self._detect_language(result)

        return f"{header}\n```{lang}\n{result.content}\n```"

    def _format_header(self, index: int, result: ReadResult) -> str:
        """Format the block header with source marker."""
        parts = [f"[S{index}]"]

        if self._config.include_file_header:
            parts.append(f"# {result.file_path}")

        # Section-specific formatting: show section title and pages
        if result.address.kind == AddressKind.SECTION:
            section_title = result.metadata.get("section_title")
            if section_title:
                parts.append(f"— {section_title}")
            page_start = result.metadata.get("page_start")
            page_end = result.metadata.get("page_end")
            if page_start is not None:
                if page_end is not None and page_end != page_start:
                    parts.append(f"(pages {page_start}-{page_end})")
                else:
                    parts.append(f"(page {page_start})")
        elif result.line_range:
            start, end = result.line_range
            parts.append(f"(lines {start}-{end})")

        kind = result.address.metadata.get("kind")
        if kind:
            parts.append(f"[{kind}]")

        context_type = result.metadata.get("context_type")
        if context_type:
            parts.append(f"({context_type})")

        return " ".join(parts)

    def _detect_language(self, result: ReadResult) -> str:
        """Detect language for syntax highlighting."""
        if result.address.kind in (AddressKind.SECTION, AddressKind.TABLE):
            return ""

        path = result.file_path.lower()
        if path.endswith(".py"):
            return "python"
        elif path.endswith((".ts", ".tsx")):
            return "typescript"
        elif path.endswith((".js", ".jsx")):
            return "javascript"
        elif path.endswith(".java"):
            return "java"
        elif path.endswith(".go"):
            return "go"
        elif path.endswith((".md", ".markdown")):
            return "markdown"
        return ""

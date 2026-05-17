# fitz_sage/engines/fitz_krag/retrieval/expander.py
"""
Code expander — enriches read results with contextual code.

Adds imports, class context (signature + __init__), and same-file references
to provide the LLM with enough context to understand the code.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.import_graph_store import ImportGraphStore
    from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
    from fitz_sage.engines.fitz_krag.ingestion.symbol_store import SymbolStore

logger = logging.getLogger(__name__)


class CodeExpander:
    """Expands read results with contextual code."""

    def __init__(
        self,
        raw_store: "RawFileStore",
        symbol_store: "SymbolStore",
        import_store: "ImportGraphStore",
        config: "FitzKragConfig",
    ):
        self._raw_store = raw_store
        self._symbol_store = symbol_store
        self._import_store = import_store
        self._config = config
        self._entity_graph_store: Any = None  # Set by engine if enrichment enabled

    def expand(
        self, read_results: list[ReadResult], entity_expansion_limit: int = 3
    ) -> list[ReadResult]:
        """
        Expand read results with context.

        For SYMBOL addresses:
        1. Add file-level imports as a header
        2. If method, add class signature + __init__
        3. Add same-file referenced symbols
        4. Add import-followed summaries

        Args:
            entity_expansion_limit: Max related chunks to pull via entity graph.
                Increase for broad/thematic queries to improve cross-document coverage.
        """
        if self._config.max_expansion_depth == 0:
            return read_results

        expanded = list(read_results)

        for result in read_results:
            if result.address.kind != AddressKind.SYMBOL:
                continue

            # 1. File imports
            expanded = self._add_file_imports(expanded, result)

            # 2. Class context for methods
            if self._config.include_class_context:
                expanded = self._add_class_context(expanded, result)

            # 3. Same-file referenced symbols
            if self._config.max_reference_expansions > 0:
                expanded = self._add_same_file_references(expanded, result)

            # 4. Import-followed summaries
            if self._config.include_import_summaries:
                expanded = self._add_import_summaries(expanded, result)

        # 5. Entity graph expansion (for all result types)
        if self._entity_graph_store:
            expanded = self._add_entity_related(expanded, read_results, entity_expansion_limit)

        return self._deduplicate(expanded)

    def _add_file_imports(self, expanded: list[ReadResult], result: ReadResult) -> list[ReadResult]:
        """Add file-level import block as context."""
        raw_file = self._raw_store.get(result.address.source_id)
        if not raw_file:
            return expanded

        lines = raw_file["content"].splitlines()
        import_lines = []
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")):
                import_lines.append(line)
            elif stripped and not stripped.startswith("#") and import_lines:
                break  # Past import section

        if not import_lines:
            return expanded

        import_text = "\n".join(import_lines)

        # Check if we already have this import block
        for existing in expanded:
            if existing.file_path == result.file_path and existing.content == import_text:
                return expanded

        import_addr = Address(
            kind=AddressKind.FILE,
            source_id=result.address.source_id,
            location=f"{result.file_path} (imports)",
            summary="File imports",
            metadata={"context_type": "imports"},
        )
        expanded.append(
            ReadResult(
                address=import_addr,
                content=import_text,
                file_path=result.file_path,
                line_range=(1, len(import_lines)),
                metadata={"context_type": "imports"},
            )
        )
        return expanded

    def _add_class_context(
        self, expanded: list[ReadResult], result: ReadResult
    ) -> list[ReadResult]:
        """Add class signature + __init__ for method symbols."""
        kind = result.address.metadata.get("kind")
        if kind != "method":
            return expanded

        qualified = result.address.metadata.get("qualified_name", "")
        parts = qualified.rsplit(".", 2)
        if len(parts) < 3:
            return expanded

        # Find the class in symbol store
        class_qualified = ".".join(parts[:-1])
        # Search by name for the class
        class_name = parts[-2]
        class_results = self._symbol_store.search_by_name(class_name, limit=5)

        for cls in class_results:
            if cls["qualified_name"] == class_qualified and cls["kind"] == "class":
                # Read the class signature (first few lines)
                raw_file = self._raw_store.get(cls["raw_file_id"])
                if not raw_file:
                    break

                lines = raw_file["content"].splitlines()
                start = cls["start_line"] - 1
                # Just the class definition line + docstring
                class_header_end = min(start + 5, cls["end_line"])
                header = "\n".join(lines[start:class_header_end])

                # Check if already present
                already = any(
                    e.content == header and e.file_path == raw_file["path"] for e in expanded
                )
                if not already:
                    class_addr = Address(
                        kind=AddressKind.SYMBOL,
                        source_id=cls["raw_file_id"],
                        location=class_qualified,
                        summary=f"Class {class_name} (context)",
                        metadata={
                            "context_type": "class_header",
                            "start_line": cls["start_line"],
                            "end_line": class_header_end,
                        },
                    )
                    expanded.append(
                        ReadResult(
                            address=class_addr,
                            content=header,
                            file_path=raw_file["path"],
                            line_range=(cls["start_line"], class_header_end),
                            metadata={"context_type": "class_header"},
                        )
                    )
                break

        return expanded

    def _add_same_file_references(
        self, expanded: list[ReadResult], result: ReadResult
    ) -> list[ReadResult]:
        """Add code of same-file symbols referenced by the current symbol."""
        symbol_id = result.address.metadata.get("symbol_id")
        if not symbol_id:
            return expanded

        # get_by_file returns references; regular get() does not
        file_symbols = self._symbol_store.get_by_file(result.address.source_id)
        if not file_symbols:
            return expanded

        # Find the current symbol to get its references list
        symbol = None
        for s in file_symbols:
            if s["id"] == symbol_id:
                symbol = s
                break
        if not symbol:
            return expanded

        references: list[str] = symbol.get("references") or []
        if not references:
            return expanded

        name_to_symbol: dict[str, dict] = {s["name"]: s for s in file_symbols}

        # Track already-present qualified names to avoid duplicates
        present = {
            r.address.metadata.get("qualified_name")
            for r in expanded
            if r.address.metadata.get("qualified_name")
        }

        added = 0
        for ref_name in references:
            if added >= self._config.max_reference_expansions:
                break

            # Strip self. prefix (e.g. "self.helper" -> "helper")
            clean_name = ref_name.split(".")[-1]
            matched = name_to_symbol.get(clean_name)
            if not matched:
                continue

            # Skip self-references
            if matched["qualified_name"] == symbol.get("qualified_name"):
                continue

            # Skip already-present
            if matched["qualified_name"] in present:
                continue

            # Read the referenced symbol's code from raw file
            raw_file = self._raw_store.get(result.address.source_id)
            if not raw_file:
                continue

            lines = raw_file["content"].splitlines()
            start = matched["start_line"] - 1
            end = matched["end_line"]
            code = "\n".join(lines[max(0, start) : end])

            ref_addr = Address(
                kind=AddressKind.SYMBOL,
                source_id=result.address.source_id,
                location=matched["qualified_name"],
                summary=f"Referenced: {matched['name']}",
                metadata={
                    "context_type": "reference",
                    "qualified_name": matched["qualified_name"],
                    "start_line": matched["start_line"],
                    "end_line": matched["end_line"],
                },
            )
            expanded.append(
                ReadResult(
                    address=ref_addr,
                    content=code,
                    file_path=result.file_path,
                    line_range=(matched["start_line"], matched["end_line"]),
                    metadata={"context_type": "reference"},
                )
            )
            present.add(matched["qualified_name"])
            added += 1

        return expanded

    def _add_import_summaries(
        self, expanded: list[ReadResult], result: ReadResult
    ) -> list[ReadResult]:
        """Add summaries of imported symbols from resolved import edges."""
        import_edges = self._import_store.get_imports(result.address.source_id)
        if not import_edges:
            return expanded

        summary_lines: list[str] = []
        seen_files: set[str] = set()

        for edge in import_edges:
            target_id = edge.get("target_file_id")
            if not target_id:
                continue  # Unresolved (stdlib/third-party) — skip
            if target_id in seen_files:
                continue
            seen_files.add(target_id)

            import_names = set(edge.get("import_names", []))
            target_symbols = self._symbol_store.get_by_file(target_id)
            if not target_symbols:
                continue

            for sym in target_symbols:
                if len(summary_lines) >= self._config.max_import_expansions:
                    break
                # Only include symbols that match import_names (if specified)
                if import_names and sym["name"] not in import_names:
                    continue
                summary_text = f"{sym['kind']} {sym['name']}"
                summary_lines.append(f"- {sym['qualified_name']}: {summary_text}")

            if len(summary_lines) >= self._config.max_import_expansions:
                break

        if not summary_lines:
            return expanded

        content = "Imported symbols:\n" + "\n".join(summary_lines)
        import_addr = Address(
            kind=AddressKind.FILE,
            source_id=result.address.source_id,
            location=f"{result.file_path} (import summaries)",
            summary="Imported symbol summaries",
            metadata={"context_type": "import_summaries"},
        )
        expanded.append(
            ReadResult(
                address=import_addr,
                content=content,
                file_path=result.file_path,
                metadata={"context_type": "import_summaries"},
            )
        )
        return expanded

    def _add_entity_related(
        self,
        expanded: list[ReadResult],
        original_results: list[ReadResult],
        max_total: int = 3,
    ) -> list[ReadResult]:
        """Find other symbols/sections sharing entities with current results."""
        try:
            # Collect IDs from original results
            chunk_ids = []
            for r in original_results:
                sid = r.address.metadata.get("symbol_id") or r.address.metadata.get("section_id")
                if sid:
                    chunk_ids.append(sid)

            if not chunk_ids:
                return expanded

            related_ids = self._entity_graph_store.get_related_chunks(
                chunk_ids, max_total=max_total
            )
            if not related_ids:
                return expanded

            # Existing IDs to avoid duplicates
            existing_ids = {
                r.address.metadata.get("symbol_id") or r.address.metadata.get("section_id")
                for r in expanded
            }

            for related_id in related_ids:
                if related_id in existing_ids:
                    continue

                # Try to fetch as symbol first, then section
                sym = self._symbol_store.get(related_id)
                if sym:
                    raw_file = self._raw_store.get(sym["raw_file_id"])
                    if raw_file:
                        lines = raw_file["content"].splitlines()
                        start = sym["start_line"] - 1
                        end = sym["end_line"]
                        code = "\n".join(lines[max(0, start) : end])
                        addr = Address(
                            kind=AddressKind.SYMBOL,
                            source_id=sym["raw_file_id"],
                            location=sym["qualified_name"],
                            summary=f"Entity-related: {sym['name']}",
                            metadata={
                                "context_type": "entity_related",
                                "symbol_id": sym["id"],
                                "qualified_name": sym["qualified_name"],
                            },
                        )
                        expanded.append(
                            ReadResult(
                                address=addr,
                                content=code,
                                file_path=raw_file["path"],
                                line_range=(sym["start_line"], sym["end_line"]),
                                metadata={"context_type": "entity_related"},
                            )
                        )
                        existing_ids.add(related_id)
        except Exception as e:
            logger.warning(f"Entity graph expansion failed: {e}")

        return expanded

    def _deduplicate(self, results: list[ReadResult]) -> list[ReadResult]:
        """Remove duplicate read results."""
        seen: set[tuple[str, str, int | None]] = set()
        deduped: list[ReadResult] = []
        for r in results:
            key = (
                r.file_path,
                r.address.location,
                r.line_range[0] if r.line_range else None,
            )
            if key not in seen:
                seen.add(key)
                deduped.append(r)
        return deduped

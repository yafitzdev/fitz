# fitz_sage/engines/fitz_krag/retrieval/reader.py
"""
Content reader — reads raw file content for addresses, extracts line ranges.

Addresses are lightweight pointers; reading fetches the actual content.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
    from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
    from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
    from fitz_sage.engines.fitz_krag.ingestion.table_store import TableStore
    from fitz_sage.tabular.store.sqlite import SqliteTableStore

logger = logging.getLogger(__name__)


def _matched_table_row_numbers(addr: Address) -> list[int]:
    """Read bounded row locations produced by concrete-value retrieval."""
    row_search = addr.metadata.get("row_search")
    if not isinstance(row_search, dict):
        return []
    values = row_search.get("row_numbers")
    if not isinstance(values, list):
        return []

    row_numbers: list[int] = []
    for value in values:
        try:
            row_number = int(value)
        except (TypeError, ValueError):
            continue
        if row_number >= 0 and row_number not in row_numbers:
            row_numbers.append(row_number)
    return row_numbers[:20]


class ContentReader:
    """Reads raw file content for addresses, extracts line ranges."""

    def __init__(
        self,
        raw_store: "RawFileStore",
        section_store: "SectionStore | None" = None,
        config: "FitzKragConfig | None" = None,
        table_store: "TableStore | None" = None,
        sqlite_table_store: "SqliteTableStore | None" = None,
        source_dir: "Path | None" = None,
    ):
        self._raw_store = raw_store
        self._section_store = section_store
        self._config = config
        self._table_store = table_store
        self._sqlite_table_store = sqlite_table_store
        self._source_dir = source_dir

    def read(self, addresses: list[Address], limit: int) -> list[ReadResult]:
        """Read content for top addresses."""
        results: list[ReadResult] = []
        for addr in addresses[:limit]:
            result = self._read_address(addr)
            if result:
                results.append(result)
        return results

    def _read_address(self, addr: Address) -> ReadResult | None:
        """Read content for a single address."""
        if addr.kind == AddressKind.SYMBOL:
            result = self._read_symbol(addr)
        elif addr.kind == AddressKind.FILE:
            result = self._read_file(addr)
        elif addr.kind == AddressKind.SECTION:
            result = self._read_section(addr)
        elif addr.kind == AddressKind.TABLE:
            result = self._read_table(addr)
        else:
            return None

        # Propagate address score so guardrails feature extractor can see it
        if result and addr.score and "score" not in result.metadata:
            result.metadata["score"] = addr.score

        return result

    def _read_symbol(self, addr: Address) -> ReadResult | None:
        """Read symbol content from raw file by line range."""
        raw_file = self._raw_store.get(addr.source_id)
        if not raw_file:
            # Disk fallback for agentic (unindexed) addresses
            content = self._read_from_disk(addr)
            if content is None:
                logger.debug(f"Raw file not found for symbol address: {addr.source_id}")
                return None
            lines = content.splitlines()
            file_path = addr.metadata.get("disk_path", addr.location)
        else:
            lines = raw_file["content"].splitlines()
            file_path = raw_file["path"]

        start = addr.metadata.get("start_line", 1) - 1  # 0-indexed
        end = addr.metadata.get("end_line", len(lines))
        code = "\n".join(lines[max(0, start) : end])

        return ReadResult(
            address=addr,
            content=code,
            file_path=file_path,
            line_range=(start + 1, end),
        )

    def _read_file(self, addr: Address) -> ReadResult | None:
        """Read entire file content."""
        raw_file = self._raw_store.get(addr.source_id)
        if not raw_file:
            # Check for pre-loaded text from agentic search (avoids re-parsing PDFs)
            text = addr.metadata.get("text")
            content = text if text else self._read_from_disk(addr)
            if content is None:
                return None
            file_path = addr.metadata.get("disk_path", addr.location)
            return ReadResult(
                address=addr,
                content=content,
                file_path=file_path,
            )

        return ReadResult(
            address=addr,
            content=raw_file["content"],
            file_path=raw_file["path"],
        )

    def _read_from_disk(self, addr: Address) -> str | None:
        """Read file content from disk when not in database (agentic path)."""
        if not self._source_dir:
            return None
        disk_path_value = addr.metadata.get("disk_path")
        if not disk_path_value:
            return None
        disk_path = str(disk_path_value)
        try:
            from fitz_sage.engines.fitz_krag.progressive.parsed_cache import (
                RICH_DOC_EXTENSIONS,
                get_parsed_text,
                parse_document_text,
            )

            full_path = self._source_dir / disk_path
            if not full_path.exists():
                return None
            ext = full_path.suffix.lower()
            if ext in RICH_DOC_EXTENSIONS:
                content_hash = addr.metadata.get("content_hash", "")
                cache_dir = getattr(self, "_cache_dir", None)
                if cache_dir and content_hash:
                    return get_parsed_text(full_path, content_hash, cache_dir)
                return parse_document_text(full_path)
            return full_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug(f"Disk read failed for {disk_path}: {e}")
        return None

    def _read_section(self, addr: Address) -> ReadResult | None:
        """Read section content from section store."""
        if not self._section_store:
            return None

        section_id = addr.metadata.get("section_id")
        if not section_id:
            return None

        section = self._section_store.get(section_id)
        if not section:
            return None

        raw_file = self._raw_store.get(addr.source_id)
        file_path = raw_file["path"] if raw_file else "unknown"

        content = section["content"]
        section_metadata = section.get("metadata")
        section_metadata = section_metadata if isinstance(section_metadata, dict) else {}
        document_title = str(section_metadata.get("document_title") or "").strip()
        metadata: dict[str, Any] = {
            "page_start": section.get("page_start"),
            "page_end": section.get("page_end"),
            "section_title": section["title"],
            "section_level": section["level"],
        }
        if document_title:
            metadata["document_title"] = document_title
            if document_title.casefold() != str(section["title"]).strip().casefold():
                content = f"[Document: {document_title}]\n{content}"

        # Add breadcrumb and child TOC when section context is enabled
        if self._config and self._config.include_section_context:
            breadcrumb = self._build_breadcrumb(section)
            if breadcrumb:
                content = f"[{breadcrumb}]\n{content}"
                metadata["breadcrumb"] = breadcrumb

            children = self._section_store.get_children(section_id)
            if children:
                child_titles = "\n".join(f"  - {c['title']}" for c in children)
                content = f"{content}\n\nSubsections:\n{child_titles}"
                metadata["child_count"] = len(children)

        return ReadResult(
            address=addr,
            content=content,
            file_path=file_path,
            metadata=metadata,
        )

    def _read_table(self, addr: Address) -> ReadResult | None:
        """Read table schema and sample data."""
        if not self._table_store:
            return None

        table_index_id = addr.metadata.get("table_index_id")
        if not table_index_id:
            return None

        record = self._table_store.get(table_index_id)
        if not record:
            return None

        table_id = record["table_id"]
        name = record["name"]
        columns = record["columns"]
        row_count = record["row_count"]

        # Get raw file for file path
        raw_file = self._raw_store.get(addr.source_id)
        file_path = raw_file["path"] if raw_file else "unknown"

        # Build schema content
        col_list = ", ".join(columns)
        content = f"Table: {name}\nColumns: {col_list}\nRow count: {row_count}"

        evidence_label = "Sample data"
        row_numbers = _matched_table_row_numbers(addr)

        # Fetch matched rows when retrieval identified them, otherwise a bounded sample.
        if self._sqlite_table_store:
            try:
                result = None
                get_rows = getattr(self._sqlite_table_store, "get_rows_by_numbers", None)
                if row_numbers and callable(get_rows):
                    result = get_rows(
                        table_id,
                        row_numbers,
                        limit=self._table_sample_limit(row_count),
                    )
                    evidence_label = "Matched data"
                if result is None:
                    table_name = self._sqlite_table_store.get_table_name(table_id)
                    col_info = self._sqlite_table_store.get_columns(table_id)
                    if table_name and col_info:
                        sanitized_cols, _ = col_info
                        cols_str = ", ".join(f'"{c}"' for c in sanitized_cols[:20])
                        sample_limit = self._table_sample_limit(row_count)
                        sql = f'SELECT {cols_str} FROM "{table_name}" LIMIT {sample_limit}'
                        result = self._sqlite_table_store.execute_query(table_id, sql)
                if result:
                    col_names, rows = result
                    if rows:
                        header = "| " + " | ".join(str(c) for c in col_names) + " |"
                        sep = "| " + " | ".join("---" for _ in col_names) + " |"
                        data_rows = []
                        for row in rows:
                            cells = [str(v)[:50] if v is not None else "" for v in row]
                            data_rows.append("| " + " | ".join(cells) + " |")
                        sample_table = "\n".join([header, sep] + data_rows)
                        content += f"\n{evidence_label}:\n{sample_table}"
            except Exception as e:
                logger.debug(f"Failed to fetch sample rows for table {table_id}: {e}")

        return ReadResult(
            address=addr,
            content=content,
            file_path=file_path,
            metadata={
                "table_id": table_id,
                "matched_row_numbers": row_numbers,
            },
        )

    def _table_sample_limit(self, row_count: int | None) -> int:
        """Return the bounded number of table rows exposed as evidence text."""
        configured = 20
        if self._config is not None:
            try:
                configured = int(getattr(self._config, "max_table_results", configured))
            except (TypeError, ValueError):
                configured = 20
        limit = min(max(1, configured), 20)
        if row_count is None:
            return limit
        try:
            count = int(row_count)
        except (TypeError, ValueError):
            return limit
        return min(limit, max(1, count))

    def _build_breadcrumb(self, section: dict[str, Any]) -> str:
        """Walk up parent_section_id chain to build a breadcrumb path.

        Caps at 5 levels to prevent runaway chains.
        """
        if not self._section_store:
            return ""

        titles: list[str] = []
        parent_id = section.get("parent_section_id")
        depth = 0

        while parent_id and depth < 5:
            parent = self._section_store.get(parent_id)
            if not parent:
                break
            titles.append(parent["title"])
            parent_id = parent.get("parent_section_id")
            depth += 1

        if not titles:
            return ""

        titles.reverse()
        return " > ".join(titles)

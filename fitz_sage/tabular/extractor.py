# fitz_sage/tabular/extractor.py
"""
Table Extractor - Extracts tables from ParsedDocument during ingestion.

Tables are extracted before chunking, converted to schema chunks with
embedded JSON data, and stored in the SQLite store for later SQL queries.
"""

from __future__ import annotations

import hashlib
import logging
from typing import TYPE_CHECKING

from fitz_sage.core.chunk import Chunk
from fitz_sage.core.document import ElementType, ParsedDocument

from .models import ParsedTable, create_schema_chunk

if TYPE_CHECKING:
    from fitz_sage.core.document import DocumentElement

logger = logging.getLogger(__name__)


class TableExtractor:
    """
    Extracts tables from ParsedDocument before chunking.

    Tables are converted to schema chunks with embedded JSON data.
    The original document is modified to remove table elements,
    so chunkers only see non-table content.

    Usage:
        extractor = TableExtractor()
        modified_doc, table_chunks = extractor.extract(parsed_doc)

        # table_chunks go to the SQLite store
        # modified_doc goes to chunker (tables removed)
    """

    def extract(self, document: ParsedDocument) -> tuple[ParsedDocument, list[Chunk]]:
        """
        Extract tables from document.

        Args:
            document: Parsed document potentially containing tables.

        Returns:
            Tuple of:
            - Modified document with table elements removed
            - List of schema chunks for extracted tables
        """
        schema_chunks: list[Chunk] = []
        non_table_elements: list[DocumentElement] = []

        for element in document.elements:
            if element.type == ElementType.TABLE:
                table = self._parse_markdown_table(element, document.source)
                if table and table.rows:
                    chunk = create_schema_chunk(table)
                    schema_chunks.append(chunk)
                    logger.debug(
                        f"Extracted table {table.table_id}: "
                        f"{table.column_count} cols, {table.row_count} rows"
                    )
                else:
                    # Keep malformed tables as regular content
                    non_table_elements.append(element)
            else:
                non_table_elements.append(element)

        if schema_chunks:
            logger.info(f"Extracted {len(schema_chunks)} tables from {document.source}")

        # Create modified document without table elements
        modified_doc = ParsedDocument(
            source=document.source,
            elements=non_table_elements,
            metadata=document.metadata,
        )

        return modified_doc, schema_chunks

    def _parse_markdown_table(self, element: DocumentElement, source: str) -> ParsedTable | None:
        """
        Parse markdown table to structured form.

        Handles standard markdown table format:
        | Header1 | Header2 |
        |---------|---------|
        | Cell1   | Cell2   |

        Args:
            element: Document element with table content.
            source: Source document path.

        Returns:
            ParsedTable if valid, None if malformed.
        """
        lines = element.content.strip().split("\n")
        if len(lines) < 2:
            return None

        # Parse header row
        headers = self._parse_row(lines[0])
        if not headers:
            return None

        # Find separator row and skip it
        # Separator looks like |---|---| or ---|--- or similar
        data_start = 1
        if len(lines) > 1 and self._is_separator_row(lines[1]):
            data_start = 2

        # Parse data rows
        rows: list[list[str]] = []
        for line in lines[data_start:]:
            cells = self._parse_row(line)
            if cells:
                # Pad or truncate to match header count
                if len(cells) < len(headers):
                    cells.extend([""] * (len(headers) - len(cells)))
                elif len(cells) > len(headers):
                    cells = cells[: len(headers)]
                rows.append(cells)

        if not rows:
            return None

        # Generate stable table_id from content hash
        table_id = hashlib.md5(element.content.encode()).hexdigest()[:12]

        return ParsedTable(
            table_id=table_id,
            source_doc=source,
            headers=headers,
            rows=rows,
            page=element.page,
        )

    def _parse_row(self, line: str) -> list[str]:
        """Parse a table row into cells."""
        # Remove leading/trailing pipes and whitespace
        line = line.strip()
        if line.startswith("|"):
            line = line[1:]
        if line.endswith("|"):
            line = line[:-1]

        # Split by pipe and strip each cell
        cells = [cell.strip() for cell in line.split("|")]
        return [c for c in cells if c]  # Remove empty cells

    def _is_separator_row(self, line: str) -> bool:
        """Check if line is a table separator (---|---|---)."""
        # Remove pipes and whitespace
        cleaned = line.replace("|", "").replace(" ", "").replace("-", "").replace(":", "")
        # Separator row should be mostly dashes, so cleaned should be empty or very short
        return len(cleaned) <= 2 and "-" in line


__all__ = ["TableExtractor"]

# fitz_sage/ingestion/chunking/plugins/table.py
"""
Table chunker for structured data files (CSV, SQLite, Excel).

Unlike regular chunkers, this doesn't split table data into chunks.
Instead, it creates a single schema description chunk for search
(e.g., "What columns does the employee table have?").

The actual table data is stored separately in TableStore (handled by executor).

Chunker ID format: "table"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from fitz_sage.core.chunk import Chunk
from fitz_sage.core.document import ParsedDocument

logger = logging.getLogger(__name__)


@dataclass
class TableChunker:
    """
    Chunker for tabular data files (CSV, SQLite, Excel).

    Creates schema description chunks for search, allowing queries like:
    - "What columns are in the employee data?"
    - "Which tables contain sales information?"
    - "What data do we have about customers?"

    The actual table data (rows) is stored in TableStore separately.

    Example:
        >>> chunker = TableChunker()
        >>> chunker.chunker_id
        'table'
        >>> chunks = chunker.chunk(parsed_csv_document)
        >>> chunks[0].content
        "Table 'employees' has 4 columns: employee_id, name, department, salary..."
    """

    plugin_name: str = field(default="table", repr=False)
    supported_extensions: List[str] = field(default_factory=lambda: [".csv", ".tsv"], repr=False)

    @property
    def chunker_id(self) -> str:
        """Unique identifier for this chunker configuration."""
        return self.plugin_name

    def chunk(self, document: ParsedDocument) -> List[Chunk]:
        """
        Create schema description chunks for table files.

        For each table in document.tables, creates ONE chunk with:
        - Schema information (column names, row count)
        - Table metadata (source file, table ID)

        This chunk enables search for table discovery:
        "What columns are in employees table?" → finds this schema chunk

        The actual table rows are NOT chunked - they go to TableStore.

        Args:
            document: ParsedDocument with tables[] populated by CSV/SQLite parser

        Returns:
            List of Chunk objects (one per table), each containing schema info
        """
        if not document.tables:
            logger.warning(f"TableChunker received document with no tables: {document.source}")
            return []

        chunks: List[Chunk] = []

        # Extract doc_id from document metadata or source path
        doc_id = document.metadata.get("doc_id")
        if not doc_id:
            source_path = Path(document.source.replace("file:///", ""))
            doc_id = source_path.stem if source_path.stem else "unknown"

        # Create one chunk per table
        for idx, table in enumerate(document.tables):
            # Build schema description for search
            schema_text = (
                f"Table '{table.id}' from {Path(table.source_file).name}\n\n"
                f"Columns ({len(table.columns)}): {', '.join(table.columns)}\n"
                f"Rows: {table.row_count}\n\n"
                f"This table can be queried using SQL."
            )

            # Metadata for chunk
            chunk_meta = {
                "source_file": table.source_file,
                "doc_id": doc_id,
                "table_id": table.id,
                "column_names": table.columns,
                "row_count": table.row_count,
                "column_count": len(table.columns),
                "is_table_schema": True,  # Mark as schema chunk for special handling
                **table.metadata,
            }

            chunk_id = f"{doc_id}:{idx}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=idx,
                    content=schema_text,
                    metadata=chunk_meta,
                )
            )

        logger.debug(f"Created {len(chunks)} schema chunks for {len(document.tables)} tables")
        return chunks


__all__ = ["TableChunker"]

# fitz_sage/ingestion/parser/plugins/csv.py
"""
CSV/TSV parser for tabular data files.

Converts CSV files into ParsedDocument with structured tables.
Tables are stored in TableStore for SQL-like querying.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass, field
from typing import Set

from fitz_sage.core.document import DocumentElement, ElementType, ParsedDocument, Table
from fitz_sage.ingestion.parser.base import ParseError
from fitz_sage.ingestion.source.base import SourceFile
from fitz_sage.logging.logger import get_logger

from .base_parser import BaseParser

logger = get_logger(__name__)


@dataclass
class CSVParser(BaseParser):
    """
    Parser for CSV and TSV files.

    Extracts structured table data (columns + rows) and creates a Table object
    in ParsedDocument.tables. Also creates a schema description element for
    search ("What columns are in the employee data?").

    Example:
        parser = CSVParser()
        doc = parser.parse(source_file)
        doc.tables[0].columns  # ['employee_id', 'name', 'department', 'salary']
    """

    plugin_name: str = field(default="csv")
    supported_extensions: Set[str] = field(default_factory=lambda: {".csv", ".tsv"})
    max_rows: int = 100000  # Safety limit for very large CSV files
    encoding: str = "utf-8"

    def parse(self, file: SourceFile) -> ParsedDocument:
        """
        Parse a CSV/TSV file into structured table data.

        Args:
            file: SourceFile with local_path for reading

        Returns:
            ParsedDocument with:
            - tables: List containing one Table with structured data
            - elements: One TABLE element with schema description
            - metadata: Table info (row count, column count, etc.)

        Raises:
            ParseError: If file is empty or malformed
        """
        file_path = file.local_path

        # Detect delimiter (CSV vs TSV)
        delimiter = "\t" if file_path.suffix.lower() == ".tsv" else ","

        # Read CSV file
        try:
            with open(file_path, "r", encoding=self.encoding, newline="", errors="replace") as f:
                reader = csv.reader(f, delimiter=delimiter)
                all_rows = list(reader)
        except Exception as e:
            raise ParseError(
                f"Failed to read CSV file: {e}",
                source=file.uri,
                cause=e,
            )

        if not all_rows:
            raise ParseError("CSV file is empty", source=file.uri)

        # Extract header and data rows
        columns = all_rows[0]
        if not columns or not any(columns):
            raise ParseError("No headers found in CSV", source=file.uri)

        data_rows = all_rows[1:]

        # Normalize rows to match column count
        normalized_rows = []
        for row in data_rows:
            if len(row) < len(columns):
                # Pad short rows
                row = row + [""] * (len(columns) - len(row))
            elif len(row) > len(columns):
                # Truncate long rows
                row = row[: len(columns)]
            normalized_rows.append(row)

        # Safety check for very large files
        if len(normalized_rows) > self.max_rows:
            logger.warning(f"CSV file has {len(normalized_rows)} rows, limiting to {self.max_rows}")
            normalized_rows = normalized_rows[: self.max_rows]

        # Create stable table ID from file path
        path_str = str(file_path.resolve())
        table_id = hashlib.md5(path_str.encode()).hexdigest()[:12]

        # Create Table object
        table = Table(
            id=table_id,
            columns=columns,
            rows=normalized_rows,
            source_file=str(file_path),
            metadata={
                "file_type": "csv" if delimiter == "," else "tsv",
                "encoding": self.encoding,
            },
        )

        # Create schema description element (for search)
        schema_text = (
            f"Table '{file_path.stem}' has {len(columns)} columns: {', '.join(columns)}. "
            f"Contains {len(normalized_rows)} rows of data from {file_path.name}."
        )

        schema_element = DocumentElement(
            type=ElementType.TABLE,
            content=schema_text,
            metadata={
                "table_id": table_id,
                "column_names": columns,
                "row_count": len(normalized_rows),
                "is_schema_description": True,
            },
        )

        logger.debug(
            f"Parsed table {file_path.name}: {len(columns)} columns, {len(normalized_rows)} rows"
        )

        # Create ParsedDocument
        return ParsedDocument(
            source=file.uri,
            elements=[schema_element],
            tables=[table],
            metadata=self._build_metadata(
                file,
                is_tabular=True,  # Flag for executor to store in TableStore
                table_id=table_id,
                column_count=len(columns),
                row_count=len(normalized_rows),
                file_type="csv" if delimiter == "," else "tsv",
            ),
        )


__all__ = ["CSVParser"]

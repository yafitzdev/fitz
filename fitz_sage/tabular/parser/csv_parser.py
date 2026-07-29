# fitz_sage/tabular/parser/csv_parser.py
"""CSV file parser for table ingestion."""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path

from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)

# File extensions this parser handles
SUPPORTED_EXTENSIONS = {".csv", ".tsv"}


@dataclass
class ParsedTableFile:
    """Result of parsing a table file."""

    table_id: str
    source_file: str
    columns: list[str]
    rows: list[list[str]]

    @property
    def row_count(self) -> int:
        """Number of data rows (excluding header)."""
        return len(self.rows)


def can_parse(file_path: Path) -> bool:
    """Check if this parser can handle the file."""
    return file_path.suffix.lower() in SUPPORTED_EXTENSIONS


def parse_csv(file_path: Path) -> ParsedTableFile:
    """
    Parse CSV/TSV file to structured table.

    Args:
        file_path: Path to the CSV or TSV file

    Returns:
        ParsedTableFile with headers and data rows

    Raises:
        ValueError: If file is empty or has no headers
    """
    suffix = file_path.suffix.lower()

    # Determine delimiter based on extension
    delimiter = "\t" if suffix == ".tsv" else ","

    # Read file with proper encoding handling
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            reader = csv.reader(f, delimiter=delimiter)
            rows = list(reader)
    except OSError as exc:
        reason = exc.strerror or type(exc).__name__
        raise ValueError(f"Failed to read table file: {reason}") from exc
    except csv.Error as exc:
        raise ValueError(f"Failed to parse table data: {exc}") from exc

    if not rows:
        raise ValueError("Empty table file")

    columns = rows[0]
    if not columns or not any(columns):
        raise ValueError("No headers found in first table row")

    data_rows = rows[1:]

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

    # Generate stable table_id from file path
    # Using relative path if possible for consistency
    path_str = str(file_path.resolve())
    table_id = hashlib.md5(path_str.encode()).hexdigest()[:12]

    logger.debug(
        f"Parsed table {file_path.name}: {len(columns)} columns, {len(normalized_rows)} rows"
    )

    return ParsedTableFile(
        table_id=table_id,
        source_file=str(file_path),
        columns=columns,
        rows=normalized_rows,
    )


def get_sample_rows(parsed: ParsedTableFile, n: int = 3) -> list[list[str]]:
    """
    Get sample rows for schema chunk content.

    Args:
        parsed: Parsed table
        n: Number of sample rows to return

    Returns:
        Up to n rows from the table
    """
    return parsed.rows[:n]

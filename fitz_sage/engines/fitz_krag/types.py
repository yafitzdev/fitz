# fitz_sage/engines/fitz_krag/types.py
"""
Core types for the Fitz KRAG engine.

Address: pointer to a knowledge unit (code symbol, file, section, table)
ReadResult: content read from an address with file path and line range
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AddressKind(str, Enum):
    """Kind of knowledge unit an address points to."""

    SYMBOL = "symbol"
    FILE = "file"
    SECTION = "section"
    TABLE = "table"


@dataclass(frozen=True)
class Address:
    """
    Pointer to a knowledge unit.

    Addresses are lightweight references used for ranking before reading.
    The summary field enables ranking without reading actual content.

    Metadata varies by kind:
    - SYMBOL: name, qualified_name, kind, start_line, end_line, signature
    - FILE: file_type, size_bytes
    - SECTION: heading, level, parent_section
    - TABLE: table_index_id, table_id, name, columns, row_count
    """

    kind: AddressKind
    source_id: str
    location: str
    summary: str
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReadResult:
    """
    Content read from an address.

    Produced by ContentReader after reading raw file content for an address.
    """

    address: Address
    content: str
    file_path: str
    line_range: tuple[int, int] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

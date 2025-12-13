# ingest/chunking/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, runtime_checkable

from core.models.chunk import Chunk


@runtime_checkable
class ChunkerPlugin(Protocol):
    """
    Protocol for ingestion chunker plugins.

    Contract:
    - Input: raw text + base metadata
    - Output: list[Chunk] (canonical core.models.chunk.Chunk)

    Plugins live in:
        ingest.chunking.plugins.<name>
    and declare:
        plugin_name: str
    """

    plugin_name: str

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        ...


@dataclass
class BaseChunker:
    """
    Convenience base class for chunker plugins.

    Concrete plugins may subclass this, but only the ChunkerPlugin contract
    is required.
    """

    plugin_name: str = "base"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

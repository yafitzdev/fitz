from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Any, Protocol

from rag.models.chunk import Chunk as _CoreChunk

# Backwards-compatible alias
Chunk = _CoreChunk


class ChunkerPlugin(Protocol):
    plugin_name: str
    """
    Protocol for ingestion chunker plugins.

    A chunker plugin takes raw text plus a base metadata dict and returns
    a list of Chunk objects. Plugins are responsible for:

    - Splitting the text into units (by characters, sentences, pages, etc.)
    - Optionally enriching metadata per chunk (e.g., page number)
    - Returning a list of universal Chunk objects.

    Plugins typically live in:
        fitz_ingest.chunker.plugins.<name>

    and declare a unique:
        plugin_name: str
    """

    # Required for auto-discovery:
    # plugin_name: str = "unique-name"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Split `text` into chunks and return a list of Chunk objects.

        The `base_meta` dict provides initial metadata (e.g. source_file).
        Plugins may copy and enrich this metadata per chunk.
        """
        ...


@dataclass
class BaseChunker:
    """
    Abstract base class for chunking plugins.

    Concrete plugins SHOULD subclass this, but do not have to. The
    auto-discovery registry only requires:

        - a class attribute `plugin_name: str`
        - a method `chunk_text(self, text, base_meta) -> List[Chunk]`

    This class exists both as documentation and as a convenient base type
    for simple chunkers.
    """

    # Optional default plugin name; concrete implementations should override
    plugin_name: str = "base"

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        raise NotImplementedError

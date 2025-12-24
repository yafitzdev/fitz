# fitz_ai/ingest/chunking/base.py
"""
Base protocol and classes for chunking plugins.

Each chunker plugin must implement:
- plugin_name: str - The plugin identifier (e.g., "simple", "markdown")
- chunker_id: str - Unique ID including params that affect output (e.g., "simple:1000:0")
- chunk_text(text, base_meta) -> List[Chunk]

The chunker_id is used for:
1. State tracking - stored per file to detect config changes
2. Re-chunking decisions - if chunker_id changes, file is re-ingested
3. Reproducibility - same chunker_id guarantees same chunking behavior
"""

from __future__ import annotations

from typing import Any, Dict, List, Protocol, runtime_checkable

from fitz_ai.engines.classic_rag.models.chunk import Chunk


@runtime_checkable
class ChunkerPlugin(Protocol):
    """
    Protocol for ingestion chunker plugins.

    Contract:
    - plugin_name: Identifies the plugin type (e.g., "simple", "markdown")
    - chunker_id: Unique ID including params that affect chunk output
    - chunk_text: Splits text into chunks with metadata

    The chunker_id format is: "{plugin_name}:{param1}:{param2}:..."
    Each plugin defines which parameters are included in its ID.

    Example:
        >>> chunker = SimpleChunker(chunk_size=1000, chunk_overlap=100)
        >>> chunker.plugin_name
        'simple'
        >>> chunker.chunker_id
        'simple:1000:100'
    """

    plugin_name: str

    @property
    def chunker_id(self) -> str:
        """
        Unique identifier including parameters that affect chunk output.

        This ID is stored in the state file per-file. When the ID changes
        (e.g., chunk_size changed from 1000 to 800), the file will be
        re-chunked on the next ingestion run.

        Format: "{plugin_name}:{param1}:{param2}:..."

        Returns:
            Deterministic string ID for this chunker configuration.
        """
        ...

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        """
        Split text into chunks.

        Args:
            text: The raw text content to chunk.
            base_meta: Base metadata to include in each chunk (source_file, doc_id, etc.)

        Returns:
            List of Chunk objects with content and metadata.
        """
        ...


__all__ = ["ChunkerPlugin", "Chunk"]

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class SimpleChunker:
    """
    Very simple text chunker used for tests and default ingestion.

    - Splits text into fixed-size character blocks.
    - Trims *outer* whitespace but keeps internal newlines.
    - Produces dict-based chunks:
        {
            "text": <chunk_text>,
            "metadata": { ... base_meta ... }
        }
    """

    plugin_name: str = "simple"
    chunk_size: int = 1000

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
        size = int(self.chunk_size)
        chunks: List[Dict[str, Any]] = []

        for i in range(0, len(text), size):
            piece = text[i:i + size]

            # Strip leading/trailing whitespace without touching internal formatting
            piece = piece.strip()

            if not piece:
                continue

            chunks.append({
                "text": piece,
                "metadata": dict(base_meta),
            })

        return chunks

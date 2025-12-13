# ingest/chunking/plugins/simple.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from core.models.chunk import Chunk


@dataclass
class SimpleChunker:
    """
    Minimal deterministic chunker.

    - Splits text into fixed-size character blocks.
    - Trims outer whitespace.
    - Emits canonical Chunk objects.
    """

    plugin_name: str = "simple"
    chunk_size: int = 1000

    def chunk_text(self, text: str, base_meta: Dict[str, Any]) -> List[Chunk]:
        size = max(1, int(self.chunk_size))

        doc_id = str(base_meta.get("doc_id") or base_meta.get("source_file") or "unknown")

        chunks: List[Chunk] = []
        chunk_index = 0

        for i in range(0, len(text), size):
            piece = text[i : i + size].strip()
            if not piece:
                continue

            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    doc_id=doc_id,
                    chunk_index=chunk_index,
                    content=piece,
                    metadata=dict(base_meta),
                )
            )
            chunk_index += 1

        return chunks

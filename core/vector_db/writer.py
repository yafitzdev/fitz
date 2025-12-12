from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Set
from uuid import uuid4
import hashlib

from rag.core import Chunk

from core.vector_db.engine import VectorDBEngine
from core.vector_db.base import VectorRecord
from core.llm.embedding.engine import EmbeddingEngine


def compute_chunk_hash(chunk: Chunk) -> str:
    """
    Stable hash based on text and identifying metadata.
    """
    m = hashlib.md5()
    m.update(chunk.text.encode("utf-8"))

    meta = chunk.metadata or {}
    m.update(str(meta.get("source", "")).encode("utf-8"))
    m.update(str(meta.get("path", "")).encode("utf-8"))

    return m.hexdigest()


@dataclass
class VectorDBWriter:
    """
    Ingestion component:
        Chunk → Embedding → VectorRecord → VectorDBEngine.upsert()

    - Adds UUIDv4 IDs when missing
    - Computes stable per-chunk hashes for dedup
    - Writes payload+vector to the vector DB
    """

    embedder: EmbeddingEngine
    vectordb: VectorDBEngine
    deduplicate: bool = True

    seen_hashes: Set[str] = field(default_factory=set, init=False)

    def write(self, collection: str, chunks: Iterable[Chunk]) -> int:
        records: List[VectorRecord] = []

        for chunk in chunks:
            h = compute_chunk_hash(chunk)

            if self.deduplicate and h in self.seen_hashes:
                continue
            self.seen_hashes.add(h)

            chunk_id = chunk.id or str(uuid4())
            vector = self.embedder.embed(chunk.text)

            payload = dict(chunk.metadata or {})
            payload["chunk_id"] = chunk_id
            payload["chunk_hash"] = h

            rec = VectorRecord(id=chunk_id, vector=vector, payload=payload)
            records.append(rec)

        if records:
            self.vectordb.upsert(collection, records)

        return len(records)

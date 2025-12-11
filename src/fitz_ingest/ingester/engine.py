from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

from fitz_ingest.chunker.engine import ChunkingEngine
from fitz_ingest.ingester.validation import IngestionValidator
from fitz_ingest.vector_db.qdrant_utils import ensure_collection

from fitz_ingest.exceptions.vector import IngestionVectorError


class IngestionEngine:
    """
    High-level ingestion coordinator.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        vector_size: int,
        chunker_engine: ChunkingEngine,
        embedder: Optional[Any] = None,
        validator: Optional[IngestionValidator] = None,
    ) -> None:
        self.client = client
        self.collection = collection
        self.vector_size = int(vector_size)
        self.chunker_engine = chunker_engine
        self.embedder = embedder
        self.validator = validator or IngestionValidator()

    # ---------------------------------------------------------
    # NEW: folder ingestion
    # ---------------------------------------------------------
    def ingest_path(self, path: str | Path) -> None:
        """
        Ingest a file or all files in a folder (recursively).
        """
        p = Path(path)

        if p.is_file():
            self.ingest_file(p)
            return

        if p.is_dir():
            for file in p.rglob("*"):
                if file.is_file():
                    self.ingest_file(file)
            return

        raise ValueError(f"Invalid ingestion path: {path}")

    # ---------------------------------------------------------
    # Single-file ingestion
    # ---------------------------------------------------------
    def ingest_file(self, path: str | Path) -> None:
        p = Path(path)

        chunks = self.chunker_engine.chunk_file(p)
        if not chunks:
            return

        self.validator.validate_chunks(chunks, str(p))

        ensure_collection(self.client, self.collection, self.vector_size)

        points = list(self._build_points(chunks))
        if not points:
            return

        # Support positional and keyword upsert
        try:
            self.client.upsert(collection_name=self.collection, points=points)
        except TypeError:
            self.client.upsert(self.collection, points)

    # ---------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------
    def _build_points(self, chunks: Iterable[Any]) -> Iterable[Dict[str, Any]]:
        for idx, ch in enumerate(chunks):
            text, metadata = self._extract_text_and_metadata(ch)

            file_val = None
            if isinstance(metadata, dict):
                file_val = metadata.get("source_file") or metadata.get("file")

            payload = dict(metadata or {})
            payload.setdefault("text", text)
            if file_val is not None:
                payload.setdefault("file", file_val)

            vector = self._embed_text(text)

            yield {
                "id": idx,
                "vector": vector,
                "payload": payload,
            }

    def _extract_text_and_metadata(self, chunk: Any) -> tuple[str, Dict[str, Any]]:
        if isinstance(chunk, dict):
            text = chunk.get("text", "")
            metadata = chunk.get("metadata", {}) or {}
            if not isinstance(metadata, dict):
                metadata = {}
            return str(text), metadata

        text = getattr(chunk, "text", "")
        metadata = getattr(chunk, "metadata", {}) or {}
        if not isinstance(metadata, dict):
            metadata = {}

        return str(text), metadata

    def _embed_text(self, text: str) -> List[float]:
        if self.embedder is None:
            return [0.0] * self.vector_size

        vec = list(self.embedder.embed(text))

        if len(vec) != self.vector_size:
            raise IngestionVectorError(
                f"Embedder returned vector of size {len(vec)} "
                f"but expected {self.vector_size}"
            )

        return vec

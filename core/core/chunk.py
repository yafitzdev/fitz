from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class Chunk:
    """
    Universal Chunk type shared across all Fitz modules:
    - fitz_stack
    - fitz_rag
    - fitz_ingest

    Fields:
        id: Optional unique identifier for the chunk.
            In ingestion this may be None until assigned.
            In retrieval this is usually the Qdrant point ID.

        text: The raw text content of this chunk.

        metadata: Arbitrary metadata dictionary.
            May include source information, file paths, tags, etc.

        score: Optional floating-point ranking signal.
            Used by rerankers and vector search retrieval.
    """
    id: Optional[str]
    text: str
    metadata: Dict[str, Any]
    score: Optional[float] = None

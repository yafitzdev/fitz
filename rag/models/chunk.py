# rag/models/chunk.py
from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field


class Chunk(BaseModel):
    """
    Canonical chunk model used across the entire fitz stack.
    """

    id: str = Field(..., description="Chunk ID")
    doc_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_index: int = Field(..., description="Order inside parent document")

    @property
    def text(self) -> str:
        """Backward-compatible alias for content."""
        return self.content

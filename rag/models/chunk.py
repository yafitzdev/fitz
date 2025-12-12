from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict


class Chunk(BaseModel):
    """
    A single chunk extracted from a document.

    Validation errors here naturally become Pydantic errors.
    These are treated as ConfigError by higher-level ingestion logic.
    """
    id: str = Field(..., description="Chunk ID.")
    doc_id: str = Field(..., description="ID of the parent document.")
    content: str = Field(..., description="Text content of the chunk.")
    metadata: Dict = Field(default_factory=dict, description="Chunk-level metadata.")
    chunk_index: int = Field(..., description="Order of the chunk inside the parent document.")

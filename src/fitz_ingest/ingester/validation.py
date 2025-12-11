# fitz_ingest/ingester/validation.py
"""
Validation utilities for fitz-ingest.

Used by the ingestion engine to ensure that, before writing to Qdrant:

- Chunks are structurally valid
- Chunk sizes are within reasonable bounds
- Metadata has required keys (if configured)

This is intentionally light-weight for v0.1.0 and can be extended later.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any, Sequence, Optional


class IngestionValidationError(Exception):
    """Raised when ingestion validation fails for a document or its chunks."""


@dataclass
class ChunkValidationConfig:
    """
    Configuration for validating chunks.

    min_chars / max_chars:
        Bounds for chunk.text length. Very loose defaults on purpose so
        we don't break existing pipelines, but still catch extreme cases.

    required_metadata_keys:
        Metadata keys that must be present in each chunk.metadata.
        For v0.1.0 this defaults to an empty tuple (no strict requirements),
        but the structure is ready to be tightened later.
    """

    min_chars: int = 1
    max_chars: int = 50_000
    required_metadata_keys: Sequence[str] = ()


class IngestionValidator:
    """
    Central ingestion validator for fitz-ingest.

    Typical usage:

        validator = IngestionValidator()
        validator.validate_chunks(chunks, file_path)
    """

    def __init__(self, config: Optional[ChunkValidationConfig] = None) -> None:
        self.config = config or ChunkValidationConfig()

    # -----------------------------------------------------
    # Public API
    # -----------------------------------------------------
    def validate_chunks(self, chunks: Iterable[Any], file_path: str | Path) -> None:
        """
        Validate all chunks for a given file.

        This does NOT raise on empty chunk lists — the ingestion engine
        already short-circuits those and they never reach Qdrant.
        """
        file_str = str(file_path)

        for idx, chunk in enumerate(chunks):
            self._validate_single_chunk(chunk, idx, file_str)

    # -----------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------
    def _validate_single_chunk(self, chunk: Any, idx: int, file_str: str) -> None:
        # Text
        text = getattr(chunk, "text", None)
        if not isinstance(text, str):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has non-string text (type={type(text)})"
            )

        length = len(text)
        if length < self.config.min_chars or length > self.config.max_chars:
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid text length {length} "
                f"(allowed range: {self.config.min_chars}-{self.config.max_chars})"
            )

        # Metadata
        metadata = getattr(chunk, "metadata", None)
        if not isinstance(metadata, Mapping):
            raise IngestionValidationError(
                f"Chunk {idx} in '{file_str}' has invalid metadata type "
                f"(expected Mapping, got {type(metadata)})"
            )

        # Required keys (configurable, empty by default)
        for key in self.config.required_metadata_keys:
            if key not in metadata:
                raise IngestionValidationError(
                    f"Chunk {idx} in '{file_str}' is missing required metadata key '{key}'"
                )

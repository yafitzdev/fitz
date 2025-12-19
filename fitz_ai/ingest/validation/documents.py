# ingest/validation/documents.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from fitz_ai.ingest.ingestion.base import RawDocument


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    min_chars: int = 1
    strip_whitespace: bool = True


def validate(
    documents: Iterable[RawDocument],
    config: ValidationConfig | None = None,
) -> List[RawDocument]:
    """
    Validate raw ingested documents before chunking.

    Rules:
    - drop empty/whitespace-only documents
    - if strip_whitespace=True, validate on stripped content and persist stripped content
    """
    cfg = config or ValidationConfig()
    valid: List[RawDocument] = []

    for doc in documents:
        content = doc.content or ""

        if cfg.strip_whitespace:
            content = content.strip()

        if len(content) < cfg.min_chars:
            continue

        if content != doc.content:
            doc = RawDocument(path=doc.path, content=content, metadata=dict(doc.metadata))

        valid.append(doc)

    return valid

from __future__ import annotations

from typing import Iterable, List
from dataclasses import dataclass
from ingest.ingester.base import RawDocument


@dataclass
class ValidationConfig:
    min_chars: int = 1
    strip_whitespace: bool = True


def validate(documents: Iterable[RawDocument], config: ValidationConfig | None = None) -> List[RawDocument]:
    """
    Validate raw ingested documents before chunking.
    Removes documents with empty or whitespace-only content.
    """
    cfg = config or ValidationConfig()
    valid: List[RawDocument] = []

    for doc in documents:
        text = doc.content or ""

        if cfg.strip_whitespace:
            text = text.strip()

        if len(text) < cfg.min_chars:
            continue

        valid.append(doc)

    return valid

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict


@dataclass
class RawDocument:
    path: str
    content: str
    metadata: Dict


class IngestPlugin:
    """
    Source-level ingestion plugin.
    """

    plugin_name: str

    def ingest(self, source: str, options: Dict) -> Iterable[RawDocument]:
        raise NotImplementedError

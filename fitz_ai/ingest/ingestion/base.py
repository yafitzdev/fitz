# ingest/ingestion/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class RawDocument:
    path: str
    content: str
    metadata: Dict[str, Any]


@runtime_checkable
class IngestPlugin(Protocol):
    """
    Source-level ingestion plugin.

    Contract:
    - ingest(source, kwargs) yields RawDocument
    - provider/plugin selection lives in config, not here
    """

    plugin_name: str

    def ingest(self, source: str, kwargs: Dict[str, Any]) -> Iterable[RawDocument]: ...

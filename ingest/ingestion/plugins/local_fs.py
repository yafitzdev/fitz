from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict

from ingest.ingestion.base import IngestPlugin, RawDocument
from ingest.ingestion.registry import register


@register
class LocalFSIngestPlugin(IngestPlugin):
    plugin_name = "local"

    def ingest(self, source: str, options: Dict) -> Iterable[RawDocument]:
        base = Path(source)
        for path in base.glob("**/*"):
            if path.is_file():
                yield RawDocument(
                    path=str(path),
                    content=path.read_text(encoding="utf-8"),
                    metadata={"source": "local_fs"},
                )

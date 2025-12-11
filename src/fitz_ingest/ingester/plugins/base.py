from typing import Iterable, Protocol
from dataclasses import dataclass

@dataclass
class RawDocument:
    path: str
    content: str
    metadata: dict

class IngestPlugin(Protocol):
    def ingest(self, source: str, config: dict) -> Iterable[RawDocument]:
        """Turn a source (folder, bucket, repo, API endpoint) into RawDocuments."""
        ...

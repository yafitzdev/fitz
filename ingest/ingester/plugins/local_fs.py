import os
from ingest.ingester.plugins.base import IngestPlugin, RawDocument

class LocalFSIngestPlugin(IngestPlugin):
    def ingest(self, source: str, config: dict):
        for root, _, files in os.walk(source):
            for f in files:
                path = os.path.join(root, f)
                try:
                    with open(path, "r", encoding="utf-8") as fh:
                        yield RawDocument(
                            path=path,
                            content=fh.read(),
                            metadata={"source": "local_fs"}
                        )
                except Exception:
                    continue

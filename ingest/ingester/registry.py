from typing import Dict, Type
from ingest.ingester.plugins.base import IngestPlugin
from ingest.ingester.plugins.local_fs import LocalFSIngestPlugin

REGISTRY: Dict[str, IngestPlugin] = {
    "local": LocalFSIngestPlugin(),
}

def get_ingest_plugin(name: str) -> IngestPlugin:
    try:
        return REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown ingest plugin: {name}")

from typing import Iterable
from ingest.ingester.registry import get_ingest_plugin
from ingest.ingester.plugins.base import RawDocument

class Ingester:
    def __init__(self, plugin_name: str, config: dict):
        self.plugin = get_ingest_plugin(plugin_name)
        self.config = config

    def run(self, source: str) -> Iterable[RawDocument]:
        return self.plugin.ingest(source, self.config)

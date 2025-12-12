# ingest/ingestion/engine.py

from __future__ import annotations

from typing import Iterable

from ingest.ingestion.base import RawDocument
from ingest.ingestion import registry
from ingest.config.schema import IngestConfig
from ingest.exceptions.config import IngestionConfigError

from core.logging.logger import get_logger
from core.logging.tags import INGEST

logger = get_logger(__name__)


class Ingester:
    """
    Ingestion engine wrapper.

    Supports:
    - plugin-style ingesters (ingest())
    - engine-style ingesters (run())
    """

    def __init__(self, *, config: IngestConfig | None = None, plugin=None, options=None):
        if config is not None:
            built = self.from_config(config)
            self.plugin = built.plugin
            self.options = built.options
            return

        self.plugin = plugin
        self.options = options or {}

    # ---------------------------------------------------------
    # Factory
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: IngestConfig) -> "Ingester":
        if not cfg.ingester or not cfg.ingester.plugin_name:
            raise IngestionConfigError("IngesterConfig.plugin_name is required")

        try:
            PluginFactory = registry.get_ingest_plugin(cfg.ingester.plugin_name)
        except Exception as e:
            raise IngestionConfigError(
                f"Unknown ingester plugin '{cfg.ingester.plugin_name}'"
            ) from e

        try:
            plugin = PluginFactory(**(cfg.ingester.options or {}))
        except Exception as e:
            raise IngestionConfigError(
                f"Failed to initialize ingester plugin '{cfg.ingester.plugin_name}'"
            ) from e

        return cls(
            plugin=plugin,
            options=cfg.ingester.options or {},
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def run(self, source: str) -> Iterable[RawDocument]:
        logger.info(f"{INGEST} Running ingestion on source={source}")

        # Engine-style ingester (used in tests)
        if hasattr(self.plugin, "run"):
            return self.plugin.run(source)

        # Plugin-style ingester (normal path)
        return self.plugin.ingest(source, self.options)

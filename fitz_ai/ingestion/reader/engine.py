# fitz_ai/ingestion/reader/engine.py
from __future__ import annotations

from typing import Any, Dict, Iterable

from fitz_ai.engines.classic_rag.config import IngestConfig
from fitz_ai.ingestion.exceptions.config import IngestionConfigError
from fitz_ai.ingestion.reader.base import IngestPlugin, RawDocument
from fitz_ai.ingestion.reader.registry import get_ingest_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import INGEST

logger = get_logger(__name__)


class IngestionEngine:
    """
    Ingestion engine wrapper.

    Architecture:
    - Always delegates to an injected IngestPlugin
    - Config-to-plugin wiring is done in from_config()
    """

    def __init__(self, plugin: IngestPlugin, kwargs: Dict[str, Any] | None = None):
        self._plugin = plugin
        self._kwargs: Dict[str, Any] = dict(kwargs or {})

    @classmethod
    def from_config(cls, cfg: IngestConfig) -> "IngestionEngine":
        if not cfg.ingester.plugin_name:
            raise IngestionConfigError("ingester.plugin_name is required")

        try:
            PluginCls = get_ingest_plugin(cfg.ingester.plugin_name)
        except Exception as exc:
            raise IngestionConfigError(
                f"Unknown ingester plugin {cfg.ingester.plugin_name!r}"
            ) from exc

        kwargs = dict(cfg.ingester.kwargs or {})

        try:
            plugin = PluginCls(**kwargs)
        except Exception as exc:
            raise IngestionConfigError(
                f"Failed to initialize ingester plugin {cfg.ingester.plugin_name!r}"
            ) from exc

        return cls(plugin=plugin, kwargs=kwargs)

    def run(self, source: str) -> Iterable[RawDocument]:
        logger.info(f"{INGEST} Running ingestion on source={source}")
        return self._plugin.ingest(source, self._kwargs)

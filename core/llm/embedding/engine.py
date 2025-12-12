# core/llm/embedding/engine.py
"""
Embedding engine wiring for Fitz LLMs.

Responsibilities:
- Read validated config
- Resolve credentials
- Instantiate embedding plugin
- Execute embedding calls
"""

from __future__ import annotations

from typing import Iterable, List

from core.logging.logger import get_logger
from core.logging.tags import EMBEDDING

from core.llm.credentials import resolve_api_key
from core.llm.embedding.registry import get_embedding_plugin
from core.config.schema import FitzConfig

logger = get_logger(__name__)


class EmbeddingEngine:
    def __init__(self, config: FitzConfig):
        self._config = config

        llm_cfg = config.llm
        provider = llm_cfg.provider

        api_key = resolve_api_key(
            provider=provider,
            config=llm_cfg.model_dump(),
        )

        logger.info(f"{EMBEDDING} Initializing embedding engine for provider '{provider}'")

        plugin_cls = get_embedding_plugin(provider)
        self._plugin = plugin_cls(
            api_key=api_key,
            model=llm_cfg.model,
        )

    def embed(self, texts: Iterable[str]) -> List[list[float]]:
        """
        Generate embeddings for a list of texts.
        """
        return self._plugin.embed(texts)

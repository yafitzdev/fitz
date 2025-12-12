# core/llm/rerank/engine.py
"""
Rerank engine wiring for Fitz LLMs.

Responsibilities:
- Read validated config
- Resolve credentials
- Instantiate rerank plugin
- Execute rerank calls
"""

from __future__ import annotations

from typing import Iterable, List

from core.logging.logger import get_logger
from core.logging.tags import RERANK

from core.llm.credentials import resolve_api_key
from core.llm.rerank.registry import get_rerank_plugin
from core.config.schema import FitzConfig

logger = get_logger(__name__)


class RerankEngine:
    def __init__(self, config: FitzConfig):
        self._config = config

        llm_cfg = config.llm
        provider = llm_cfg.provider

        api_key = resolve_api_key(
            provider=provider,
            config=llm_cfg.model_dump(),
        )

        logger.info(f"{RERANK} Initializing rerank engine for provider '{provider}'")

        plugin_cls = get_rerank_plugin(provider)
        self._plugin = plugin_cls(
            api_key=api_key,
            model=llm_cfg.model,
        )

    def rerank(
        self,
        query: str,
        documents: Iterable[str],
        top_k: int | None = None,
    ) -> List[int]:
        """
        Rerank documents for a query.

        Returns
        -------
        List[int]
            Indices of documents sorted by relevance (best first)
        """
        return self._plugin.rerank(
            query=query,
            documents=documents,
            top_k=top_k,
        )

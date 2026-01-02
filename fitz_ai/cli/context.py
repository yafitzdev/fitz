# fitz_ai/cli/context.py
"""
Central CLI context - single source of truth for all CLI commands.

All configuration reading happens here. Commands import CLIContext and use it.
No more scattered dict.get() chains across every command file.

Usage:
    from fitz_ai.cli.context import CLIContext

    # Load config (returns None if no config exists)
    ctx = CLIContext.load()
    if ctx:
        print(ctx.chat_plugin)        # always exists
        print(ctx.chat_display)       # "cohere (command-r-plus)"
        print(ctx.retrieval_top_k)    # always exists
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


@dataclass
class CLIContext:
    """
    Central CLI context containing all configuration values.

    This is the single source of truth for CLI commands. All config values
    are extracted once and exposed as typed properties.
    """

    # Raw and typed config
    raw_config: dict = field(repr=False)
    typed_config: Any = field(default=None, repr=False)
    config_path: Optional[Path] = field(default=None)

    # Chat LLM
    chat_plugin: str = ""
    chat_model_smart: str = ""
    chat_model_fast: str = ""

    # Embedding
    embedding_plugin: str = ""
    embedding_model: str = ""

    # Vector DB
    vector_db_plugin: str = ""
    vector_db_kwargs: dict = field(default_factory=dict)

    # Retrieval
    retrieval_plugin: str = ""
    retrieval_collection: str = ""
    retrieval_top_k: int = 5

    # Rerank
    rerank_enabled: bool = False
    rerank_plugin: str = ""
    rerank_model: str = ""

    # RGS (Response Generation System)
    rgs_citations: bool = True
    rgs_strict_grounding: bool = True

    # -------------------------------------------------------------------------
    # Display Properties (formatted strings for UI)
    # -------------------------------------------------------------------------

    @property
    def chat_display(self) -> str:
        """Chat info for display: 'plugin (model)' or just 'plugin'."""
        if self.chat_model_smart:
            return f"{self.chat_plugin} ({self.chat_model_smart})"
        return self.chat_plugin or "?"

    @property
    def chat_display_fast(self) -> str:
        """Chat fast model info for display."""
        if self.chat_model_fast:
            return f"{self.chat_plugin} ({self.chat_model_fast})"
        return self.chat_plugin or "?"

    @property
    def embedding_display(self) -> str:
        """Embedding info for display: 'plugin (model)' or just 'plugin'."""
        if self.embedding_model:
            return f"{self.embedding_plugin} ({self.embedding_model})"
        return self.embedding_plugin or "?"

    @property
    def rerank_display(self) -> Optional[str]:
        """Rerank info for display, or None if disabled."""
        if not self.rerank_enabled:
            return None
        if self.rerank_model:
            return f"{self.rerank_plugin} ({self.rerank_model})"
        return self.rerank_plugin or "?"

    @property
    def vector_db_display(self) -> str:
        """Vector DB info for display."""
        host = self.vector_db_kwargs.get("host", "")
        port = self.vector_db_kwargs.get("port", "")
        if host:
            return f"{self.vector_db_plugin} ({host}:{port})"
        return self.vector_db_plugin or "?"

    @property
    def retrieval_display(self) -> str:
        """Retrieval info for display."""
        return f"{self.retrieval_plugin} (collection={self.retrieval_collection}, top_k={self.retrieval_top_k})"

    @property
    def embedding_id(self) -> str:
        """Unique embedding ID for caching (e.g., 'cohere:embed-english-v3.0')."""
        plugin = self.embedding_plugin or "unknown"
        model = self.embedding_model or "default"
        return f"{plugin}:{model}"

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls) -> Optional["CLIContext"]:
        """
        Load CLI context from user config.

        Returns None if no config exists (user hasn't run 'fitz init').

        Returns:
            CLIContext with all values populated, or None if no config.
        """
        from fitz_ai.core.config import ConfigNotFoundError, load_config_dict
        from fitz_ai.engines.fitz_rag.config import FitzRagConfig

        # Determine config path (engine-specific first, then global)
        config_path = FitzPaths.engine_config("fitz_rag")
        if not config_path.exists():
            config_path = FitzPaths.config()

        # Load raw config
        try:
            raw_config = load_config_dict(config_path)
        except ConfigNotFoundError:
            return None
        except Exception:
            return None

        # Parse into typed config
        try:
            typed_config = FitzRagConfig.from_dict(raw_config)
        except Exception:
            typed_config = None

        return cls._from_config(raw_config, typed_config, config_path)

    @classmethod
    def _from_config(cls, config: dict, typed: Any, path: Optional[Path]) -> "CLIContext":
        """
        Extract all values from config + plugin defaults.

        Config provides plugin names. Plugin YAMLs provide model defaults.
        User kwargs override plugin defaults.
        """
        from fitz_ai.llm.loader import load_plugin

        # Get config sections with defaults
        chat = config.get("chat", {})
        emb = config.get("embedding", {})
        vdb = config.get("vector_db", {})
        ret = config.get("retrieval", {})
        rerank = config.get("rerank", {})
        rgs = config.get("rgs", {})

        # Extract plugin names
        chat_plugin_name = chat.get("plugin_name", "")
        emb_plugin_name = emb.get("plugin_name", "")
        rerank_plugin_name = rerank.get("plugin_name", "")

        # Load plugin specs to get model defaults
        chat_models = {}
        if chat_plugin_name:
            try:
                chat_spec = load_plugin("chat", chat_plugin_name)
                chat_kwargs = chat.get("kwargs", {})
                chat_models = chat_kwargs.get("models", chat_spec.defaults.get("models", {}))
            except Exception:
                chat_models = chat.get("kwargs", {}).get("models", {})

        emb_model = ""
        if emb_plugin_name:
            try:
                emb_spec = load_plugin("embedding", emb_plugin_name)
                emb_kwargs = emb.get("kwargs", {})
                emb_model = emb_kwargs.get("model", emb_spec.defaults.get("model", ""))
            except Exception:
                emb_model = emb.get("kwargs", {}).get("model", "")

        rerank_model = ""
        if rerank.get("enabled") and rerank_plugin_name:
            try:
                rerank_spec = load_plugin("rerank", rerank_plugin_name)
                rerank_kwargs = rerank.get("kwargs", {})
                rerank_model = rerank_kwargs.get("model", rerank_spec.defaults.get("model", ""))
            except Exception:
                rerank_model = rerank.get("kwargs", {}).get("model", "")

        return cls(
            raw_config=config,
            typed_config=typed,
            config_path=path,
            # Chat
            chat_plugin=chat_plugin_name,
            chat_model_smart=chat_models.get("smart", ""),
            chat_model_fast=chat_models.get("fast", ""),
            # Embedding
            embedding_plugin=emb_plugin_name,
            embedding_model=emb_model,
            # Vector DB
            vector_db_plugin=vdb.get("plugin_name", ""),
            vector_db_kwargs=vdb.get("kwargs", {}),
            # Retrieval
            retrieval_plugin=ret.get("plugin_name", ""),
            retrieval_collection=ret.get("collection", "default"),
            retrieval_top_k=ret.get("top_k", 5),
            # Rerank
            rerank_enabled=rerank.get("enabled", False),
            rerank_plugin=rerank_plugin_name,
            rerank_model=rerank_model,
            # RGS
            rgs_citations=rgs.get("enable_citations", True),
            rgs_strict_grounding=rgs.get("strict_grounding", True),
        )

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def get_vector_db_client(self):
        """Get vector DB client instance."""
        from fitz_ai.vector_db.registry import get_vector_db_plugin

        return get_vector_db_plugin(self.vector_db_plugin, **self.vector_db_kwargs)

    def get_collections(self) -> list[str]:
        """Get list of collections from vector DB."""
        try:
            client = self.get_vector_db_client()
            return sorted(client.list_collections())
        except Exception:
            return []

    def info_line(self, include_rerank: bool = True) -> str:
        """
        Get a single-line info string for display.

        Example: "Collection: default | VectorDB: local_faiss | Chat: cohere (command-r-plus)"
        """
        parts = [
            f"Collection: {self.retrieval_collection}",
            f"VectorDB: {self.vector_db_plugin}",
            f"Retrieval: {self.retrieval_plugin}",
            f"Chat: {self.chat_display}",
            f"Embedding: {self.embedding_display}",
        ]
        if include_rerank and self.rerank_display:
            parts.append(f"Rerank: {self.rerank_display}")
        return " | ".join(parts)


__all__ = ["CLIContext"]

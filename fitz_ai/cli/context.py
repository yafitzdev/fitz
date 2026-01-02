# fitz_ai/cli/context.py
"""
Central CLI context - single source of truth for all CLI commands.

All configuration reading happens here. Commands import CLIContext and use it.
No more scattered dict.get() chains across every command file.

Config Loading Strategy:
    1. Package defaults (fitz_ai/engines/<engine>/config/default.yaml) - always loaded
    2. User config (.fitz/config/<engine>.yaml) - overrides defaults

Values are ALWAYS guaranteed to exist. No fallback logic needed in CLI code.

Usage:
    from fitz_ai.cli.context import CLIContext

    # Load merged config (defaults + user overrides) - always succeeds
    ctx = CLIContext.load()
    print(ctx.chat_plugin)        # always exists
    print(ctx.chat_display)       # "cohere (command-r-plus)"
    print(ctx.retrieval_top_k)    # always exists

    # Check if user has customized config
    if ctx.has_user_config:
        print(f"Using config from {ctx.config_path}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from fitz_ai.config.loader import get_config_source, load_engine_config
from fitz_ai.core.paths import FitzPaths
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)

DEFAULT_ENGINE = "fitz_rag"


@dataclass
class CLIContext:
    """
    Central CLI context containing all configuration values.

    This is the single source of truth for CLI commands. All config values
    are extracted once and exposed as typed properties.

    The context is ALWAYS valid - package defaults guarantee all values exist.
    """

    # Raw and typed config
    raw_config: dict = field(repr=False)
    typed_config: Any = field(default=None, repr=False)
    config_path: Path = field(default=None)
    config_source: str = field(default="")
    has_user_config: bool = field(default=False)

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
    def load(cls, engine: str = DEFAULT_ENGINE) -> "CLIContext":
        """
        Load CLI context with merged config (defaults + user overrides).

        This ALWAYS succeeds because package defaults always exist.
        Values are guaranteed to be present - no fallback logic needed.

        Args:
            engine: Engine name (default: "fitz_rag")

        Returns:
            CLIContext with all values populated from merged config.
        """
        # Load merged config (defaults + user overrides)
        merged_config = load_engine_config(engine)

        # Get config source info
        source = get_config_source(engine)
        user_config_path = FitzPaths.engine_config(engine)
        has_user = user_config_path.exists()

        # Load typed config if possible
        typed_config = cls._load_typed_config(merged_config, engine)

        # Extract all values from merged config
        return cls._from_config(
            config=merged_config,
            typed=typed_config,
            path=user_config_path if has_user else None,
            source=source,
            has_user=has_user,
        )

    @classmethod
    def _load_typed_config(cls, config: dict, engine: str) -> Any:
        """Load typed config from raw config dict."""
        if engine == "fitz_rag":
            try:
                from fitz_ai.engines.fitz_rag.config import FitzRagConfig

                return FitzRagConfig.from_dict(config)
            except Exception as e:
                logger.debug(f"Could not load typed config: {e}")
                return None
        return None

    @classmethod
    def _from_config(
        cls,
        config: dict,
        typed: Any,
        path: Optional[Path],
        source: str,
        has_user: bool,
    ) -> "CLIContext":
        """
        Extract all values from merged config.

        Since config is merged (defaults + user), all values are guaranteed to exist.
        No .get() fallbacks needed - values come from defaults if not overridden.
        """
        # Get config sections - guaranteed to exist from defaults
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
        chat_models = cls._get_chat_models(chat, chat_plugin_name)
        emb_model = cls._get_embedding_model(emb, emb_plugin_name)
        rerank_model = cls._get_rerank_model(rerank, rerank_plugin_name)

        return cls(
            raw_config=config,
            typed_config=typed,
            config_path=path,
            config_source=source,
            has_user_config=has_user,
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

    @staticmethod
    def _get_chat_models(chat: dict, plugin_name: str) -> dict:
        """Get chat model names from config or plugin defaults."""
        from fitz_ai.llm.loader import load_plugin

        if not plugin_name:
            return {}

        chat_kwargs = chat.get("kwargs", {})

        # User-specified models take precedence
        if "models" in chat_kwargs:
            return chat_kwargs["models"]

        # Fall back to plugin defaults
        try:
            spec = load_plugin("chat", plugin_name)
            return spec.defaults.get("models", {})
        except Exception:
            return {}

    @staticmethod
    def _get_embedding_model(emb: dict, plugin_name: str) -> str:
        """Get embedding model name from config or plugin defaults."""
        from fitz_ai.llm.loader import load_plugin

        if not plugin_name:
            return ""

        emb_kwargs = emb.get("kwargs", {})

        # User-specified model takes precedence
        if "model" in emb_kwargs:
            return emb_kwargs["model"]

        # Fall back to plugin defaults
        try:
            spec = load_plugin("embedding", plugin_name)
            return spec.defaults.get("model", "")
        except Exception:
            return ""

    @staticmethod
    def _get_rerank_model(rerank: dict, plugin_name: str) -> str:
        """Get rerank model name from config or plugin defaults."""
        from fitz_ai.llm.loader import load_plugin

        if not rerank.get("enabled") or not plugin_name:
            return ""

        rerank_kwargs = rerank.get("kwargs", {})

        # User-specified model takes precedence
        if "model" in rerank_kwargs:
            return rerank_kwargs["model"]

        # Fall back to plugin defaults
        try:
            spec = load_plugin("rerank", plugin_name)
            return spec.defaults.get("model", "")
        except Exception:
            return ""

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

    def require_collections(self) -> list[str]:
        """
        Get collections or exit with error if none exist.

        Use this in commands that require at least one collection to exist.
        """
        import typer

        from fitz_ai.cli.ui import ui

        collections = self.get_collections()
        if not collections:
            ui.error("No collections found. Run 'fitz ingest' first.")
            raise typer.Exit(1)
        return collections

    def require_typed_config(self):
        """
        Get typed config or exit with error if invalid.

        Use this in commands that require a valid typed config (e.g., fitz_rag pipeline).
        """
        import typer

        from fitz_ai.cli.ui import ui

        if self.typed_config is None:
            ui.error("Invalid config. Run 'fitz init' to reconfigure.")
            raise typer.Exit(1)
        return self.typed_config

    def require_vector_db_client(self):
        """
        Get vector DB client or exit with error if connection fails.

        Use this in commands that require a working vector DB connection.
        """
        import typer

        from fitz_ai.cli.ui import ui

        try:
            return self.get_vector_db_client()
        except Exception as e:
            ui.error(f"Failed to connect to vector DB '{self.vector_db_plugin}': {e}")
            raise typer.Exit(1)

    def select_collection(self, collection: Optional[str] = None, *, require: bool = True) -> str:
        """
        Select a collection interactively or use the provided one.

        This method keeps ctx.retrieval_collection and typed_config in sync.

        Args:
            collection: Pre-selected collection name (skips prompt if valid)
            require: If True, exit with error if no collections exist

        Returns:
            Selected collection name
        """
        from fitz_ai.cli.ui import ui

        # If collection provided, validate and use it
        if collection:
            self.retrieval_collection = collection
            if self.typed_config is not None:
                self.typed_config.retrieval.collection = collection
            return collection

        # Get available collections
        if require:
            collections = self.require_collections()
        else:
            collections = self.get_collections()
            if not collections:
                return self.retrieval_collection

        # Auto-select if only one
        if len(collections) == 1:
            selected = collections[0]
            ui.info(f"Using collection: {selected}")
        else:
            print()
            selected = ui.prompt_numbered_choice(
                "Collection", collections, self.retrieval_collection
            )

        # Keep ctx and typed_config in sync
        self.retrieval_collection = selected
        if self.typed_config is not None:
            self.typed_config.retrieval.collection = selected

        return selected

    def select_engine(self, engine: Optional[str] = None) -> str:
        """
        Select an engine interactively or use the provided one.

        Args:
            engine: Pre-selected engine name (skips prompt if valid)

        Returns:
            Selected engine name
        """
        import typer

        from fitz_ai.cli.ui import ui
        from fitz_ai.runtime import get_default_engine, get_engine_registry, list_engines

        available = list_engines()
        registry = get_engine_registry()

        if engine is None:
            print()
            descriptions = registry.list_with_descriptions()
            default = get_default_engine()
            return ui.prompt_engine_selection(
                engines=available,
                descriptions=descriptions,
                default=default,
            )

        if engine not in available:
            ui.error(f"Unknown engine: '{engine}'. Available: {', '.join(available)}")
            raise typer.Exit(1)

        return engine

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


__all__ = ["CLIContext", "DEFAULT_ENGINE"]

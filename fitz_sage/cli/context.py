# fitz_sage/cli/context.py
"""
Central CLI context - single source of truth for all CLI commands.

All configuration reading happens here. Commands import CLIContext and use it.
No more scattered dict.get() chains across every command file.

Config Loading Strategy:
    1. Package defaults (fitz_sage/engines/<engine>/config/default.yaml) - always loaded
    2. User config (.fitz/config.yaml) - overrides defaults

Values are ALWAYS guaranteed to exist. No fallback logic needed in CLI code.

Usage:
    from fitz_sage.cli.context import CLIContext

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

from fitz_sage.config.loader import get_config_source, load_engine_config
from fitz_sage.core.paths import FitzPaths
from fitz_sage.logging.logger import get_logger

logger = get_logger(__name__)


def _default_engine() -> str:
    from fitz_sage.runtime import get_default_engine

    return get_default_engine()


@dataclass
class CLIContext:
    """
    Central CLI context containing all configuration values.

    This is the single source of truth for CLI commands. All config values
    are extracted once and exposed as typed properties.

    The context is ALWAYS valid - package defaults guarantee all values exist.
    """

    # Engine
    engine_name: str = ""

    # Raw and typed config
    raw_config: dict = field(default_factory=dict, repr=False)
    typed_config: Any = field(default=None, repr=False)
    config_path: Path = field(default=None)
    config_source: str = field(default="")
    has_user_config: bool = field(default=False)

    # Chat LLM
    chat_plugin: str = ""
    chat_model_smart: str = ""
    chat_model_fast: str = ""

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

    # Chunking/Parser
    parser: str = "docling"
    chunk_size: int = 512
    chunk_overlap: int = 0

    # Tier specs (for factory functions)
    chat_tier_specs: dict = field(default_factory=dict)

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
    def rerank_display(self) -> Optional[str]:
        """Rerank info for display, or None if disabled."""
        if not self.rerank_enabled:
            return None
        if self.rerank_model:
            return f"{self.rerank_plugin} ({self.rerank_model})"
        return self.rerank_plugin or "?"

    @property
    def retrieval_display(self) -> str:
        """Retrieval info for display."""
        return f"{self.retrieval_plugin} (collection={self.retrieval_collection}, top_k={self.retrieval_top_k})"

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def load(cls, engine: str | None = None) -> "CLIContext":
        """
        Load CLI context with merged config (defaults + user overrides).

        This ALWAYS succeeds because package defaults always exist.
        Values are guaranteed to be present - no fallback logic needed.

        Args:
            engine: Engine name (default: from user config or runtime default)

        Returns:
            CLIContext with all values populated from merged config.
        """
        if engine is None:
            engine = _default_engine()

        # Load merged config (defaults + user overrides) - returns Pydantic model
        typed_config = load_engine_config(engine)

        # Convert to dict for _from_config
        config_dict = typed_config.model_dump()

        # Get config source info
        source = get_config_source(engine)
        user_config_path = FitzPaths.config()
        has_user = user_config_path.exists()

        # Extract all values from config dict
        return cls._from_config(
            engine=engine,
            config=config_dict,
            typed=typed_config,
            path=user_config_path if has_user else None,
            source=source,
            has_user=has_user,
        )

    @classmethod
    def _load_typed_config(cls, config: dict, engine: str) -> Any:
        """Load typed config from raw config dict."""
        try:
            from fitz_sage.config.loader import load_engine_config

            return load_engine_config(engine)
        except Exception as e:
            logger.debug(f"Could not load typed config for {engine}: {e}")
            return None

    @classmethod
    def _from_config(
        cls,
        engine: str,
        config: dict,
        typed: Any,
        path: Optional[Path],
        source: str,
        has_user: bool,
    ) -> "CLIContext":
        """
        Extract all values from merged config.

        Flat config: chat_fast/chat_balanced/chat_smart as top-level provider/model specs.
        """
        # Extract provider names from flat tier specs
        chat_smart_spec = config.get("chat_smart", "")
        chat_fast_spec = config.get("chat_fast", "")
        chat_balanced_spec = config.get("chat_balanced", "")
        chat_plugin_name = cls._parse_plugin_string(chat_smart_spec)

        rerank_spec = config.get("rerank")
        rerank_plugin_name = cls._parse_plugin_string(rerank_spec)
        rerank_model = rerank_spec.split("/", 1)[1] if rerank_spec and "/" in rerank_spec else ""

        # Extract model names from tier specs
        chat_model_smart = chat_smart_spec.split("/", 1)[1] if "/" in chat_smart_spec else ""
        chat_model_fast = chat_fast_spec.split("/", 1)[1] if "/" in chat_fast_spec else ""

        # Build tier specs dict for factory
        chat_tier_specs = {
            "fast": chat_fast_spec,
            "balanced": chat_balanced_spec,
            "smart": chat_smart_spec,
        }

        return cls(
            engine_name=engine,
            raw_config=config,
            typed_config=typed,
            config_path=path,
            config_source=source,
            has_user_config=has_user,
            # Chat
            chat_plugin=chat_plugin_name,
            chat_model_smart=chat_model_smart,
            chat_model_fast=chat_model_fast,
            # Retrieval
            retrieval_plugin=config.get("retrieval_plugin", "dense"),
            retrieval_collection=config.get("collection", "default"),
            retrieval_top_k=config.get("top_k", 5),
            # Rerank
            rerank_enabled=config.get("rerank") is not None,
            rerank_plugin=rerank_plugin_name,
            rerank_model=rerank_model,
            # RGS
            rgs_citations=config.get("enable_citations", True),
            rgs_strict_grounding=config.get("strict_grounding", True),
            # Chunking/Parser
            parser=config.get("parser", "docling"),
            chunk_size=config.get("chunk_size", 512),
            chunk_overlap=config.get("chunk_overlap", 0),
            # Tier specs
            chat_tier_specs=chat_tier_specs,
        )

    @staticmethod
    def _to_dict(obj: Any) -> dict:
        """Convert PluginKwargs or dict to plain dict."""
        if obj is None:
            return {}
        if hasattr(obj, "model_dump"):
            # Pydantic model - convert and filter None values
            return {k: v for k, v in obj.model_dump().items() if v is not None}
        if isinstance(obj, dict):
            return obj
        return {}

    @staticmethod
    def _parse_plugin_string(plugin: str | None) -> str:
        """
        Parse plugin string to get provider name.

        V2 format: "provider" or "provider/model"
        Returns just the provider name.
        """
        if not plugin:
            return ""
        # Split on "/" and take first part
        return plugin.split("/")[0]

    # -------------------------------------------------------------------------
    # Component Accessors
    # -------------------------------------------------------------------------

    def get_chat_factory(self):
        """Get chat factory for per-task tier selection."""
        from fitz_sage.llm import get_chat_factory

        return get_chat_factory(self.chat_tier_specs)

    def get_chat(self, tier: str = "smart"):
        """Get chat client for specified tier."""
        factory = self.get_chat_factory()
        return factory(tier)

    def get_reranker(self):
        """Get reranker instance (None if not enabled)."""
        if not self.rerank_plugin:
            return None
        from fitz_sage.llm import get_reranker

        return get_reranker(self.rerank_plugin)

    def require_chat_factory(self):
        """Get chat factory or exit with error."""
        import typer

        from fitz_sage.cli.ui import ui

        try:
            return self.get_chat_factory()
        except Exception as e:
            ui.error(f"Failed to initialize chat: {e}")
            raise typer.Exit(1)

    def get_pipeline(self):
        """Get engine instance for the current engine."""
        if self.typed_config is None:
            raise ValueError("No typed config available")

        from fitz_sage.runtime import create_engine

        return create_engine(self.engine_name, config=self.typed_config)

    def require_pipeline(self):
        """Get engine instance or exit with error."""
        import typer

        from fitz_sage.cli.ui import ui

        # First ensure we have a valid typed config
        self.require_typed_config()

        try:
            return self.get_pipeline()
        except Exception as e:
            ui.error(f"Failed to initialize pipeline: {e}")
            raise typer.Exit(1)

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def require_typed_config(self):
        """
        Get typed config or exit with error if invalid.

        Use this in commands that require a valid typed config.
        """
        import typer

        from fitz_sage.cli.ui import ui

        if self.typed_config is None:
            ui.error("Invalid config. Edit .fitz/config.yaml to fix.")
            raise typer.Exit(1)
        return self.typed_config

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
        from fitz_sage.cli.ui import ui

        # If collection provided, validate and use it
        if collection:
            self.retrieval_collection = collection
            if self.typed_config is not None:
                self.typed_config.collection = collection  # V2: flat structure
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
            self.typed_config.collection = selected  # V2: flat structure

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

        from fitz_sage.cli.ui import ui
        from fitz_sage.runtime import get_default_engine, get_engine_registry, list_engines

        available = list_engines()
        registry = get_engine_registry()

        if engine is None:
            # Skip selection if only one engine available
            if len(available) == 1:
                return available[0]

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

        Example: "Collection: default | VectorDB: pgvector | Chat: cohere (command-r-plus)"
        """
        parts = [
            f"Collection: {self.retrieval_collection}",
            f"Retrieval: {self.retrieval_plugin}",
            f"Chat: {self.chat_display}",
        ]
        if include_rerank and self.rerank_display:
            parts.append(f"Rerank: {self.rerank_display}")
        return " | ".join(parts)


__all__ = ["CLIContext"]

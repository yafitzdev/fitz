# fitz_sage/engines/fitz_krag/runtime.py
"""
Fitz KRAG Runtime - Engine registration and entry points.

Auto-registers the fitz_krag engine with the global registry on import.
"""

from typing import Optional

from fitz_sage.config import load_engine_config
from fitz_sage.core import Answer
from fitz_sage.engines.fitz_krag.config import FitzKragConfig


def run_fitz_krag(
    query: str,
    config: Optional[FitzKragConfig] = None,
    config_path: Optional[str] = None,
) -> Answer:
    """
    Execute a Fitz KRAG query.

    Args:
        query: The question text
        config: Optional pre-loaded FitzKragConfig
        config_path: Optional path to config file

    Returns:
        Answer object with generated text and source provenance
    """
    if config is None:
        if config_path is None:
            config = load_engine_config("fitz_krag")
        else:
            from pathlib import Path

            import yaml

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_krag" in raw:
                config_dict = raw["fitz_krag"]
            else:
                config_dict = raw

            config = FitzKragConfig(**config_dict)

    from fitz_sage.core import Query
    from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

    engine = FitzKragEngine(config)
    return engine.answer(Query(text=query))


run = run_fitz_krag


# =============================================================================
# AUTO-REGISTRATION WITH GLOBAL REGISTRY
# =============================================================================


def _register_fitz_krag_engine():
    """Register Fitz KRAG engine with the global registry."""
    from fitz_sage.engines.fitz_krag.config import get_default_config_path
    from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
    from fitz_sage.runtime.registry import EngineCapabilities, EngineRegistry

    def fitz_krag_factory(config):
        """Factory for creating Fitz KRAG engine."""
        if config is None:
            config = load_engine_config("fitz_krag")
        elif isinstance(config, dict):
            config = FitzKragConfig(**config)

        return FitzKragEngine(config)

    def fitz_krag_config_loader(config_path):
        """Load config for fitz_krag engine."""
        if config_path is None:
            return load_engine_config("fitz_krag")
        else:
            from pathlib import Path

            import yaml

            with Path(config_path).open("r", encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

            if "fitz_krag" in raw:
                config_dict = raw["fitz_krag"]
            else:
                config_dict = raw

            return FitzKragConfig(**config_dict)

    def fitz_krag_list_collections():
        """List KRAG collections — scan SQLite storage dir + manifest dir."""
        import logging

        from fitz_sage.storage.sqlite import get_connection_manager

        logger = logging.getLogger(__name__)
        try:
            manager = get_connection_manager()
            if not manager._started:
                manager.start()

            collections: list[str] = []
            for collection_name in manager.list_collections():
                try:
                    with manager.connection(collection_name) as conn:
                        has_krag = conn.execute(
                            "SELECT 1 FROM sqlite_master "
                            "WHERE type='table' AND name='krag_raw_files' LIMIT 1"
                        ).fetchone()
                        if has_krag:
                            collections.append(collection_name)
                except Exception:
                    pass

            # Also discover collections with manifests (pointed but not yet fully indexed)
            try:
                from fitz_sage.core.paths import FitzPaths

                collections_dir = FitzPaths.workspace() / "collections"
                if collections_dir.exists():
                    for child in collections_dir.iterdir():
                        if child.is_dir() and (child / "manifest.json").exists():
                            if child.name not in collections:
                                collections.append(child.name)
            except Exception:
                pass

            return sorted(collections)
        except Exception as e:
            logger.warning(f"Failed to list KRAG collections: {e}")
            return []

    capabilities = EngineCapabilities(
        supports_collections=True,
        requires_documents_at_query=False,
        supports_persistent_ingest=True,
        supports_chat=True,
        supports_streaming=False,
        requires_config=True,
        requires_api_key=True,
        api_key_env_var="COHERE_API_KEY",
    )

    try:
        registry = EngineRegistry.get_global()
        registry.register(
            name="fitz_krag",
            factory=fitz_krag_factory,
            description="Knowledge Routing Augmented Generation (code + docs)",
            config_type=FitzKragConfig,
            config_loader=fitz_krag_config_loader,
            default_config_path=get_default_config_path,
            list_collections=fitz_krag_list_collections,
            capabilities=capabilities,
        )
    except ValueError:
        pass  # Already registered


# Auto-register on import
_register_fitz_krag_engine()

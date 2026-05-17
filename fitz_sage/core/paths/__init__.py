# fitz_sage/core/paths/__init__.py
"""
Central path management for Fitz.

ALL components that need file paths should use this module.
No hardcoded paths anywhere else in the codebase.

Usage:
    from fitz_sage.core.paths import FitzPaths

    config_path = FitzPaths.config()

    # Override workspace for testing
    FitzPaths.set_workspace("/tmp/test_fitz")
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import cache as _cache
from . import config as _config
from . import ingestion as _ingestion
from . import plugins as _plugins
from . import storage as _storage
from .workspace import WorkspaceManager


class FitzPaths:
    """
    Central path management for Fitz.

    Facade class that delegates to domain-specific modules.
    All methods are classmethods for static access.
    """

    # Workspace management - delegate to WorkspaceManager
    @classmethod
    def set_workspace(cls, path: Optional[str | Path]) -> None:
        """Override the workspace root."""
        WorkspaceManager.set_workspace(path)

    @classmethod
    def reset(cls) -> None:
        """Reset to default workspace (CWD)."""
        WorkspaceManager.reset()

    @classmethod
    def workspace(cls) -> Path:
        """The .fitz workspace directory."""
        return WorkspaceManager.workspace()

    @classmethod
    def ensure_workspace(cls) -> Path:
        """Get workspace path and create it if it doesn't exist."""
        return WorkspaceManager.ensure_workspace()

    # Configuration
    @classmethod
    def config(cls) -> Path:
        """Default config file path: {workspace}/config.yaml"""
        return _config.config()

    # PostgreSQL data (pgserver)
    @classmethod
    def pgdata(cls, collection: Optional[str] = None) -> Path:
        """PostgreSQL data directory for pgserver."""
        return _storage.pgdata(collection)

    @classmethod
    def ensure_pgdata(cls, collection: Optional[str] = None) -> Path:
        """Get pgdata path and create it if it doesn't exist."""
        return _storage.ensure_pgdata(collection)

    # Ingestion
    @classmethod
    def ingest_state(cls) -> Path:
        """Ingestion state file for incremental ingestion."""
        return _ingestion.ingest_state()

    @classmethod
    def table_registry(cls, collection: str) -> Path:
        """Table chunk IDs registry for a collection."""
        return _ingestion.table_registry(collection)

    @classmethod
    def ensure_table_registry_dir(cls) -> Path:
        """Get tables directory and create it if it doesn't exist."""
        return _ingestion.ensure_table_registry_dir()

    # Cache
    @classmethod
    def cache(cls) -> Path:
        """Cache directory root."""
        return _cache.cache()

    @classmethod
    def knowledge_map(cls) -> Path:
        """Knowledge map state file for visualization caching."""
        return _cache.knowledge_map()

    @classmethod
    def knowledge_map_html(cls) -> Path:
        """Default output path for knowledge map HTML visualization."""
        return _cache.knowledge_map_html()

    # User Plugins
    @classmethod
    def user_home(cls) -> Path:
        """User's fitz home directory: ~/.fitz/"""
        return _plugins.user_home()

    @classmethod
    def user_plugins(cls) -> Path:
        """User plugins directory: ~/.fitz/plugins/"""
        return _plugins.user_plugins()

    @classmethod
    def user_llm_plugins(cls, plugin_type: str) -> Path:
        """User LLM plugins: ~/.fitz/plugins/llm/{plugin_type}/"""
        return _plugins.user_llm_plugins(plugin_type)

    @classmethod
    def user_chunking_plugins(cls) -> Path:
        """User chunking plugins: ~/.fitz/plugins/chunking/"""
        return _plugins.user_chunking_plugins()

    @classmethod
    def user_constraint_plugins(cls) -> Path:
        """User constraint plugins: ~/.fitz/plugins/constraint/"""
        return _plugins.user_constraint_plugins()

    @classmethod
    def ensure_user_plugins(cls) -> Path:
        """Create user plugins directory structure if it doesn't exist."""
        return _plugins.ensure_user_plugins()


# Convenience functions
def get_workspace() -> Path:
    """Convenience function to get workspace path."""
    return FitzPaths.workspace()


def get_config_path() -> Path:
    """Convenience function to get config path."""
    return FitzPaths.config()


def get_ingest_state_path() -> Path:
    """Convenience function to get ingest state path."""
    return FitzPaths.ingest_state()


__all__ = [
    "FitzPaths",
    "get_workspace",
    "get_config_path",
    "get_ingest_state_path",
]

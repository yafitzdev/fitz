# fitz/core/paths.py
"""
Central path management for Fitz.

ALL components that need file paths should use this module.
No hardcoded paths anywhere else in the codebase.

Design principles:
- Single source of truth
- Workspace-relative by default (CWD/.fitz/)
- Easy to override for testing
- No magic, explicit paths
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


class FitzPaths:
    """
    Central path management for Fitz.

    All paths are relative to the workspace root, which defaults to
    the current working directory. This can be overridden for testing
    or special deployment scenarios.

    Usage:
        from fitz.core.paths import FitzPaths

        # Get paths
        config_path = FitzPaths.config()
        vector_db_path = FitzPaths.vector_db()

        # Override workspace for testing
        FitzPaths.set_workspace("/tmp/test_fitz")
    """

    _workspace_override: Optional[Path] = None

    @classmethod
    def set_workspace(cls, path: Optional[str | Path]) -> None:
        """
        Override the workspace root.

        Useful for testing or running multiple isolated instances.
        Pass None to reset to default (CWD).

        Args:
            path: New workspace root, or None to reset
        """
        if path is None:
            cls._workspace_override = None
        else:
            cls._workspace_override = Path(path)

    @classmethod
    def reset(cls) -> None:
        """Reset to default workspace (CWD). Useful in tests."""
        cls._workspace_override = None

    # =========================================================================
    # Core Paths
    # =========================================================================

    @classmethod
    def workspace(cls) -> Path:
        """
        The .fitz workspace directory.

        Default: {CWD}/.fitz/

        This is the root for all Fitz data in a project.
        """
        if cls._workspace_override is not None:
            return cls._workspace_override
        return Path.cwd() / ".fitz"

    @classmethod
    def ensure_workspace(cls) -> Path:
        """Get workspace path and create it if it doesn't exist."""
        path = cls.workspace()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Configuration
    # =========================================================================

    @classmethod
    def config(cls) -> Path:
        """
        Default config file path.

        Location: {workspace}/config.yaml
        """
        return cls.workspace() / "config.yaml"

    @classmethod
    def config_dir(cls) -> Path:
        """
        Config directory for multiple config files.

        Location: {workspace}/config/
        """
        return cls.workspace() / "config"

    # =========================================================================
    # Vector Database
    # =========================================================================

    @classmethod
    def vector_db(cls, collection: Optional[str] = None) -> Path:
        """
        Vector database storage directory.

        Location: {workspace}/vector_db/
        Or with collection: {workspace}/vector_db/{collection}/

        Args:
            collection: Optional collection name for collection-specific path
        """
        base = cls.workspace() / "vector_db"
        if collection:
            return base / collection
        return base

    @classmethod
    def ensure_vector_db(cls, collection: Optional[str] = None) -> Path:
        """Get vector DB path and create it if it doesn't exist."""
        path = cls.vector_db(collection)
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Documents & Ingestion
    # =========================================================================

    @classmethod
    def uploads(cls) -> Path:
        """
        Directory for uploaded/ingested documents.

        Location: {workspace}/uploads/
        """
        return cls.workspace() / "uploads"

    @classmethod
    def chunks_cache(cls) -> Path:
        """
        Cache for processed chunks (optional optimization).

        Location: {workspace}/cache/chunks/
        """
        return cls.workspace() / "cache" / "chunks"

    # =========================================================================
    # Logs & Diagnostics
    # =========================================================================

    @classmethod
    def logs(cls) -> Path:
        """
        Log files directory.

        Location: {workspace}/logs/
        """
        return cls.workspace() / "logs"

    @classmethod
    def ensure_logs(cls) -> Path:
        """Get logs path and create it if it doesn't exist."""
        path = cls.logs()
        path.mkdir(parents=True, exist_ok=True)
        return path

    # =========================================================================
    # Quickstart / Demo
    # =========================================================================

    @classmethod
    def quickstart_docs(cls) -> Path:
        """
        Sample documents created by quickstart.

        Location: {workspace}/quickstart_docs/
        """
        return cls.workspace() / "quickstart_docs"

    @classmethod
    def quickstart_config(cls) -> Path:
        """
        Config file created by quickstart.

        Location: {workspace}/quickstart_config.yaml
        """
        return cls.workspace() / "quickstart_config.yaml"


# =============================================================================
# Convenience Functions
# =============================================================================


def get_workspace() -> Path:
    """Convenience function to get workspace path."""
    return FitzPaths.workspace()


def get_vector_db_path(collection: Optional[str] = None) -> Path:
    """Convenience function to get vector DB path."""
    return FitzPaths.vector_db(collection)


def get_config_path() -> Path:
    """Convenience function to get config path."""
    return FitzPaths.config()

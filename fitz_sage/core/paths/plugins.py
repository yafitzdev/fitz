# fitz_sage/core/paths/plugins.py
"""User plugin paths (home directory)."""

from __future__ import annotations

from pathlib import Path


def user_home() -> Path:
    """
    User's fitz home directory.

    Location: ~/.fitz/
    """
    return Path.home() / ".fitz"


def user_plugins() -> Path:
    """
    User plugins directory.

    Location: ~/.fitz/plugins/
    """
    return user_home() / "plugins"


def user_llm_plugins(plugin_type: str) -> Path:
    """
    User LLM plugins directory for a specific type.

    Location: ~/.fitz/plugins/llm/{plugin_type}/

    Args:
        plugin_type: One of 'chat', 'embedding', 'rerank'
    """
    return user_plugins() / "llm" / plugin_type


def user_chunking_plugins() -> Path:
    """
    User chunking plugins directory.

    Location: ~/.fitz/plugins/chunking/
    """
    return user_plugins() / "chunking"


def user_constraint_plugins() -> Path:
    """
    User constraint plugins directory.

    Location: ~/.fitz/plugins/constraint/
    """
    return user_plugins() / "constraint"


def ensure_user_plugins() -> Path:
    """Create user plugins directory structure if it doesn't exist."""
    base = user_plugins()
    (base / "llm" / "chat").mkdir(parents=True, exist_ok=True)
    (base / "llm" / "rerank").mkdir(parents=True, exist_ok=True)
    (base / "chunking").mkdir(parents=True, exist_ok=True)
    (base / "constraint").mkdir(parents=True, exist_ok=True)
    return base

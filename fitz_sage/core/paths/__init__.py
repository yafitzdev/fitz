"""Central project workspace and managed-model path access."""

from __future__ import annotations

from pathlib import Path

from . import config as _config
from . import plugins as _plugins
from .workspace import WorkspaceManager


class FitzPaths:
    """Stable path facade used by runtime components."""

    @classmethod
    def set_workspace(cls, path: str | Path | None) -> None:
        WorkspaceManager.set_workspace(path)

    @classmethod
    def reset(cls) -> None:
        WorkspaceManager.reset()

    @classmethod
    def workspace(cls) -> Path:
        return WorkspaceManager.workspace()

    @classmethod
    def ensure_workspace(cls) -> Path:
        return WorkspaceManager.ensure_workspace()

    @classmethod
    def config(cls) -> Path:
        return _config.config()

    @classmethod
    def user_home(cls) -> Path:
        return _plugins.user_home()


def get_workspace() -> Path:
    return FitzPaths.workspace()


def get_config_path() -> Path:
    return FitzPaths.config()


__all__ = ["FitzPaths", "get_config_path", "get_workspace"]

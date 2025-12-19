# fitz/core/config/normalize.py
from __future__ import annotations

from typing import Any


def _as_plugin_config(block: dict[str, Any], *, key: str) -> dict[str, Any]:
    """
    Convert a human-facing preset block into a EnginePluginConfig dict.
    Expected human-facing shape:
      { "plugin": "<name>", "kwargs": { ... }? }
    """
    if not isinstance(block, dict):
        raise TypeError(f"Preset block '{key}' must be a mapping")

    if "plugin" not in block:
        raise ValueError(f"Preset block '{key}' missing required key: 'plugin'")

    plugin = block["plugin"]
    if not isinstance(plugin, str) or not plugin:
        raise TypeError(f"Preset block '{key}.plugin' must be a non-empty string")

    kwargs = block.get("kwargs", {})
    if kwargs is None:
        kwargs = {}
    if not isinstance(kwargs, dict):
        raise TypeError(f"Preset block '{key}.kwargs' must be a mapping")

    return {"plugin_name": plugin, "kwargs": dict(kwargs)}


def normalize_preset(preset: dict[str, Any]) -> dict[str, Any]:
    """
    Convert human-facing preset config into engine-facing runtime config.

    Human-facing expected shape (minimal):
      llm:
        chat: { plugin: ... }
        embedding: { plugin: ... }
        rerank: { plugin: ... }?   (optional)
      vector_db: { plugin: ... }
      pipeline: { plugin: ... }
    """
    if not isinstance(preset, dict):
        raise TypeError("Preset must be a mapping")

    if "llm" not in preset or not isinstance(preset["llm"], dict):
        raise ValueError("Preset missing required mapping: 'llm'")

    llm = preset["llm"]

    if "chat" not in llm:
        raise ValueError("Preset.llm missing required mapping: 'chat'")
    if "embedding" not in llm:
        raise ValueError("Preset.llm missing required mapping: 'embedding'")

    out: dict[str, Any] = {}

    out["chat"] = _as_plugin_config(llm["chat"], key="llm.chat")
    out["embedding"] = _as_plugin_config(llm["embedding"], key="llm.embedding")

    if "rerank" in llm and llm["rerank"] is not None:
        out["rerank"] = _as_plugin_config(llm["rerank"], key="llm.rerank")
    else:
        out["rerank"] = None

    if "vector_db" not in preset:
        raise ValueError("Preset missing required mapping: 'vector_db'")
    out["vector_db"] = _as_plugin_config(preset["vector_db"], key="vector_db")

    if "pipeline" not in preset:
        raise ValueError("Preset missing required mapping: 'pipeline'")
    out["pipeline"] = _as_plugin_config(preset["pipeline"], key="pipeline")

    return out

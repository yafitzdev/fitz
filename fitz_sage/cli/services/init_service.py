# fitz_sage/cli/services/init_service.py
"""Service layer for init command - handles configuration generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SystemStatus:
    """System detection results."""

    api_keys: dict[str, Any]
    llm_endpoint: Any


@dataclass
class ConfigResult:
    """Result of config generation."""

    global_config: str
    engine_config: str | None
    engine_name: str


class InitService:
    """
    Business logic for init command.

    Handles system detection, plugin filtering, and config generation
    without UI concerns. Can be used by CLI, SDK, or API.
    """

    def detect_system(self) -> SystemStatus:
        """Detect available services and API keys."""
        from fitz_sage.core.detect import detect_system_status

        system = detect_system_status()
        return SystemStatus(
            api_keys=system.api_keys,
            llm_endpoint=system.llm_endpoint,
        )

    def load_default_config(self, engine: str = "fitz_krag") -> dict:
        """Load the default configuration from package."""
        defaults_path = (
            Path(__file__).parent.parent.parent / "engines" / engine / "config" / "default.yaml"
        )
        with defaults_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def filter_available_plugins(
        self, plugins: list[str], plugin_type: str, system: SystemStatus
    ) -> list[str]:
        """
        Filter plugins to only those that are available.

        With the single-protocol LLM architecture, plugin filtering is
        much simpler: ``endpoint`` is always available (the user
        provides a URL); ``openai`` / ``azure_openai`` require an
        API key.
        """
        available = []

        for plugin in plugins:
            plugin_lower = plugin.lower()

            # endpoint is always selectable — the user supplies the URL.
            # enterprise is also always selectable; failure happens at
            # config-validation time if auth fields are missing.
            if plugin_lower in ("endpoint", "enterprise"):
                available.append(plugin)
                continue

            # OpenAI / Azure presets require OPENAI_API_KEY
            if "openai" in plugin_lower or "azure" in plugin_lower:
                if system.api_keys.get(
                    "openai", type("", (), {"available": False})
                ).available:
                    available.append(plugin)
                continue

            # Always-available local utilities (parsers, etc.).
            if plugin_lower in ("simple", "markdown", "semantic", "docling"):
                available.append(plugin)
                continue

        return available

    def get_default_model(self, plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
        """Get default model for a provider."""
        from fitz_sage.llm import get_chat, get_reranker

        try:
            if plugin_type == "chat":
                instance = get_chat(plugin_name, tier=tier)  # type: ignore[arg-type]
            elif plugin_type == "rerank":
                instance = get_reranker(plugin_name)
                if instance is None:
                    return ""
            else:
                return ""
            return getattr(instance, "_model", "")
        except Exception:
            return ""

    def generate_global_config(self, default_engine: str) -> str:
        """Generate global config YAML."""
        config = {"runtime": {"default_engine": default_engine}}
        return yaml.dump(config, sort_keys=False, default_flow_style=False)

    def copy_engine_default_config(self, engine_name: str, registry: Any) -> str | None:
        """Copy engine's default config if it exists."""
        engine_meta = registry.get_metadata(engine_name)
        if not engine_meta or not engine_meta.config_path:
            return None

        config_path = Path(engine_meta.config_path)
        if not config_path.exists():
            return None

        with config_path.open("r", encoding="utf-8") as f:
            return f.read()

    def write_config(self, global_config: str, engine_config: str | None, engine_name: str) -> None:
        """Write configuration files to disk."""
        from fitz_sage.core.paths import FitzPaths

        paths = FitzPaths()

        # Write global config
        paths.global_config.parent.mkdir(parents=True, exist_ok=True)
        paths.global_config.write_text(global_config, encoding="utf-8")

        # Write engine config if provided
        if engine_config:
            engine_config_path = paths.config_dir / f"{engine_name}.yaml"
            engine_config_path.write_text(engine_config, encoding="utf-8")

    def validate_config(self, engine_name: str) -> bool:
        """Validate generated configuration by attempting to load it."""
        try:
            from fitz_sage.runtime import get_engine_registry

            registry = get_engine_registry()
            registry.create_engine(engine_name)
            return True
        except Exception:
            return False

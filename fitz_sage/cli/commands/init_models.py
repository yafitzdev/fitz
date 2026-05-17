# fitz_sage/cli/commands/init_models.py
"""
Model selection helpers for init wizard.

Provides default model lookups and interactive model prompts.
"""

from __future__ import annotations

from fitz_sage.cli.ui import get_first_available, ui

# Default models by plugin type and provider
MODEL_DEFAULTS = {
    "chat_smart": {
        "cohere": "command-a-03-2025",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "ollama": "llama3.2",
    },
    "chat_fast": {
        "cohere": "command-r7b-12-2024",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-3-5-20241022",
        "ollama": "llama3.2:1b",
    },
    "chat_balanced": {
        "cohere": "command-r-08-2024",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-sonnet-4-20250514",
        "ollama": "llama3.2",
    },
    "rerank": {
        "cohere": "rerank-v3.5",
        "ollama": "qllama/bge-reranker-v2-m3",
    },
    "vision": {
        "cohere": "command-a-vision-07-2025",
        "openai": "gpt-4o",
        "anthropic": "claude-sonnet-4-20250514",
        "ollama": "llama3.2-vision",
    },
}


def get_default_model(plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
    """Get the default model for a plugin.

    Args:
        plugin_type: Type of plugin (chat, rerank, vision)
        plugin_name: Name of the plugin (cohere, openai, etc.)
        tier: Model tier for chat plugins ("smart", "fast", or "balanced")

    Returns:
        Default model name, or empty string if not found.
    """
    if plugin_type == "chat":
        key = f"chat_{tier}"
        return MODEL_DEFAULTS.get(key, {}).get(plugin_name, "")
    return MODEL_DEFAULTS.get(plugin_type, {}).get(plugin_name, "")


def prompt_model(plugin_type: str, plugin_name: str, tier: str = "smart") -> str:
    """Prompt for model selection with smart default.

    Args:
        plugin_type: Type of plugin (chat, rerank)
        plugin_name: Name of the plugin
        tier: Model tier for chat plugins ("smart", "fast", or "balanced")

    Returns:
        Selected model name.
    """
    default_model = get_default_model(plugin_type, plugin_name, tier)

    if not default_model:
        return ""

    if plugin_type == "chat":
        return ui.prompt_text(f"  {tier.capitalize()} model for {plugin_name}", default_model)

    return ui.prompt_text(f"  Model for {plugin_name}", default_model)


def get_default_or_first(choices: list[str], default: str) -> str:
    """Get default if available in choices, otherwise return first available.

    Args:
        choices: List of available choices
        default: Preferred default value

    Returns:
        Default if in choices, otherwise first choice.
    """
    if default in choices:
        return default
    return get_first_available(choices, default)

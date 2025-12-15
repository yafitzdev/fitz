"""
Configuration presets for Fitz.

Provides pre-configured setups for common use cases:
- local: Offline development with Ollama
- dev: Development with cloud APIs (cost-effective models)
- production: Production-ready with best models

Usage:
    from fitz.core.config.presets import get_preset

    config = get_preset("local")
    # Returns dict ready to use with pipeline
"""

# Available configuration presets
PRESETS = {
    "local": {
        "llm": {"plugin_name": "local"},
        "embedding": {"plugin_name": "local"},
        "rerank": {"plugin_name": "local"},
    },
    "dev": {
        "llm": {"plugin_name": "openai", "kwargs": {"model": "gpt-4o-mini"}},
        "embedding": {"plugin_name": "openai"},
        "rerank": {"plugin_name": "cohere"},
    },
    "production": {
        "llm": {"plugin_name": "anthropic"},
        "embedding": {"plugin_name": "cohere"},
        "rerank": {"plugin_name": "cohere"},
    },
}


def get_preset(name: str) -> dict:
    """
    Get a configuration preset by name.

    Args:
        name: Preset name ("local", "dev", or "production")

    Returns:
        Configuration dictionary ready to use with RAG pipeline

    Raises:
        ValueError: If preset name is not recognized

    Examples:
        >>> config = get_preset("local")
        >>> config["llm"]["plugin_name"]
        'local'

        >>> config = get_preset("dev")
        >>> config["llm"]["kwargs"]["model"]
        'gpt-4o-mini'

        >>> get_preset("invalid")
        Traceback (most recent call last):
        ...
        ValueError: Unknown preset: 'invalid'. Available presets: local, dev, production
    """
    if name not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(
            f"Unknown preset: {name!r}. Available presets: {available}"
        )

    # Return a deep copy to prevent mutation of the original preset
    import copy
    return copy.deepcopy(PRESETS[name])
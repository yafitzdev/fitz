# fitz_sage/llm/factory.py
"""
Chat factory for per-task tier selection.

Enables different LLM tiers for different micro-tasks within a single module.
Tier assignments are developer decisions, not user-configurable.

Usage:
    from fitz_sage.llm.factory import ChatFactory, get_chat_factory

    # Create factory once (flat config: one spec per tier)
    chat_factory = get_chat_factory({
        "fast": "endpoint/qwen2.5-3b-instruct",
        "balanced": "endpoint/qwen2.5-7b-instruct",
        "smart": "endpoint/qwen2.5-32b-instruct",
    }, {"base_url": "http://localhost:8080/v1"})

    # Use different tiers per task
    fast_chat = chat_factory("fast")       # Simple tasks
    balanced_chat = chat_factory("balanced")  # Complex tasks
    smart_chat = chat_factory("smart")     # User-facing responses

Tier Guidelines:
    - fast: Background tasks, enrichment, simple classification
    - balanced: SQL generation, structured output, bulk evaluation
    - smart: User-facing responses, complex reasoning
"""

from __future__ import annotations

from typing import Any, Callable, Literal

from fitz_sage.llm.config import create_chat_provider
from fitz_sage.llm.providers.base import ChatProvider

ModelTier = Literal["fast", "balanced", "smart"]
ChatFactory = Callable[[ModelTier], ChatProvider]


def get_chat_factory(
    tier_specs: dict[str, str | None],
    config: dict[str, Any] | None = None,
) -> ChatFactory:
    """
    Create a factory that returns cached chat clients per tier.

    Args:
        tier_specs: Dict mapping tier name to provider spec
                    (e.g., {"fast": "endpoint/qwen2.5-3b", "smart": "endpoint/qwen2.5-32b"}).
        config: Optional config with auth/base_url settings.

    Returns:
        Factory function: (tier) -> ChatProvider

    Example:
        factory = get_chat_factory(
            {"fast": "endpoint/qwen2.5-3b", "smart": "endpoint/qwen2.5-32b"},
            {"base_url": "http://localhost:8080/v1"},
        )
        chat = factory("fast")  # Returns cached fast-tier client
        chat.chat([{"role": "user", "content": "Hello"}])
    """
    configured_tiers = {tier: spec for tier, spec in tier_specs.items() if spec}
    if not configured_tiers:
        raise ValueError("At least one chat tier provider must be configured.")

    fallback_spec = (
        configured_tiers.get("fast")
        or configured_tiers.get("balanced")
        or configured_tiers.get("smart")
        or next(iter(configured_tiers.values()))
    )
    cache: dict[ModelTier, ChatProvider] = {}

    def factory(tier: ModelTier = "fast") -> ChatProvider:
        if tier not in cache:
            spec = configured_tiers.get(tier) or fallback_spec
            cache[tier] = create_chat_provider(spec, config, tier)
        return cache[tier]

    return factory


__all__ = [
    "ModelTier",
    "ChatFactory",
    "get_chat_factory",
]

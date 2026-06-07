# tests/e2e_krag/config.py
"""E2E test configuration loader.

Uses the centralized tests/test_config.yaml (same as all other tests).
"""

from __future__ import annotations

from typing import Any

from tests.conftest import load_test_config


def load_e2e_config() -> dict[str, Any]:
    """Load test configuration (same as all other tests)."""
    return load_test_config()


def get_tier_names(config: dict[str, Any] | None = None) -> list[str]:
    """Get list of tier names in order."""
    if config is None:
        config = load_test_config()
    return [tier["name"] for tier in config.get("tiers", [])]


def get_tier_config(tier_name: str, base_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Build full config dict for a specific tier.

    Args:
        tier_name: Name of the tier (e.g., "local", "cloud")
        base_config: Base config (loaded if not provided)

    Returns:
        Config dict with chat endpoint settings for the tier
    """
    if base_config is None:
        base_config = load_test_config()

    tiers = base_config.get("tiers", [])
    tier = next((t for t in tiers if t["name"] == tier_name), None)
    if not tier:
        available = [t["name"] for t in tiers]
        raise ValueError(f"Unknown tier: {tier_name}. Available: {available}")

    # Build tier config matching the expected structure. Chat carries the
    # endpoint provider, per-tier base_url, and optional api_key_env.
    return {
        "chat": {
            "plugin_name": tier["chat"],
            "base_url": tier.get("chat_base_url"),
            "api_key_env": tier.get("chat_api_key_env"),
            "models": tier.get("chat_models", {}),
        },
    }


def get_cache_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Get cache configuration."""
    if config is None:
        config = load_test_config()
    return config.get("cache", {"enabled": False, "max_entries": 1000, "ttl_days": 30})

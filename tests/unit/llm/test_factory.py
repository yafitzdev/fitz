# tests/unit/llm/test_factory.py
"""Tests for tiered chat factory helpers."""

from unittest.mock import Mock, patch

import pytest

from fitz_sage.llm.factory import get_chat_factory


def test_get_chat_factory_rejects_empty_tiers() -> None:
    """Empty tier specs fail at factory creation time."""
    with pytest.raises(ValueError, match="At least one chat tier provider"):
        get_chat_factory({})


def test_get_chat_factory_uses_configured_tier_as_fallback() -> None:
    """A single configured tier can satisfy other tier requests."""
    chat = Mock()

    with patch("fitz_sage.llm.factory.create_chat_provider", return_value=chat) as create:
        factory = get_chat_factory(
            {"smart": "endpoint/qwen2.5-32b"},
            {"base_url": "http://localhost:8080/v1"},
        )

        assert factory("fast") is chat

    create.assert_called_once_with(
        "endpoint/qwen2.5-32b",
        {"base_url": "http://localhost:8080/v1"},
        "fast",
    )

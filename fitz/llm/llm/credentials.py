# core/llm/credentials.py
"""
Centralized credential resolution for LLM providers.

Rules:
- Plugins must NOT read environment variables directly.
- Resolution order:
  1. Explicit config value
  2. Provider-specific env var
  3. Generic fallback env var
- Fail with actionable errors.
"""

from __future__ import annotations

import os
from typing import Mapping

from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import CHAT

logger = get_logger(__name__)


# Universal fallback (lowest priority)
GENERIC_API_KEY_ENV = "FITZ_LLM_API_KEY"

# Provider-specific env vars
PROVIDER_ENV_MAP: dict[str, list[str]] = {
    "openai": ["OPENAI_API_KEY"],
    "azure_openai": ["AZURE_OPENAI_API_KEY", "OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "cohere": ["COHERE_API_KEY"],
    "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
    # openai-compatible providers intentionally reuse OPENAI semantics
    "openai_compatible": ["OPENAI_API_KEY"],
}


class CredentialError(RuntimeError):
    pass


def resolve_api_key(
    *,
    provider: str,
    config: Mapping[str, str] | None = None,
) -> str:
    """
    Resolve API key for a given LLM provider.

    Parameters
    ----------
    provider:
        Logical provider name (e.g. "openai", "anthropic").
    config:
        Optional config mapping. If it contains "api_key",
        that value is used first.

    Returns
    -------
    str
        Resolved API key.

    Raises
    ------
    CredentialError
        If no API key could be resolved.
    """
    cfg = config or {}

    # 1. Explicit config
    api_key = cfg.get("api_key")
    if api_key:
        logger.debug(f"{CHAT} Using API key from explicit config for provider '{provider}'")
        return api_key

    # 2. Provider-specific env vars
    env_vars = PROVIDER_ENV_MAP.get(provider, [])
    for env_name in env_vars:
        value = os.getenv(env_name)
        if value:
            logger.debug(f"{CHAT} Using API key from env '{env_name}' for provider '{provider}'")
            return value

    # 3. Generic fallback
    fallback = os.getenv(GENERIC_API_KEY_ENV)
    if fallback:
        logger.debug(
            f"{CHAT} Using API key from env '{GENERIC_API_KEY_ENV}' for provider '{provider}'"
        )
        return fallback

    # Failure
    expected = env_vars + [GENERIC_API_KEY_ENV]
    expected_str = ", ".join(expected)

    raise CredentialError(
        f"API key for provider '{provider}' not found. "
        f"Set one of: {expected_str}, or provide 'api_key' in config."
    )

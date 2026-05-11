# fitz_sage/core/detect.py
"""
Centralized service detection for Fitz.

Auto-discovers:

- A local OpenAI-compatible HTTP server on the standard ports (the
  ``endpoint`` provider's primary use case — llama.cpp's
  ``llama-server``, vLLM, LM Studio, or Ollama in ``/v1/`` mode).
- The ``OPENAI_API_KEY`` environment variable (the ``openai`` preset).

Used by:

- ``fitz config`` / ``fitz doctor`` (status display).
- ``fitz init`` (first-run wizard recommends a default config).
"""

from __future__ import annotations

import logging
import os
import socket
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class ServiceStatus:
    """Status of a detected HTTP service."""

    name: str
    available: bool
    host: Optional[str] = None
    port: Optional[int] = None
    base_url: Optional[str] = None
    details: str = ""


@dataclass
class ApiKeyStatus:
    """Status of an API key environment variable."""

    name: str
    available: bool
    env_var: str = ""
    details: str = ""


@dataclass
class SystemStatus:
    """Complete system status."""

    llm_endpoint: ServiceStatus
    api_keys: dict[str, ApiKeyStatus]

    @property
    def best_llm_spec(self) -> str:
        """
        Recommend a chat spec given what's available.

        Priority:
          1. A local OpenAI-compatible server → ``endpoint/<placeholder>``.
          2. ``OPENAI_API_KEY`` set → ``openai/gpt-4o``.
          3. Fallback → ``endpoint/qwen2.5-7b-instruct`` (won't work
             until user starts a llama-server, but the error message
             from the endpoint provider is actionable).
        """
        if self.llm_endpoint.available:
            return "endpoint/qwen2.5-7b-instruct"
        if self.api_keys.get("openai", ApiKeyStatus(name="OpenAI", available=False)).available:
            return "openai/gpt-4o"
        return "endpoint/qwen2.5-7b-instruct"

    @property
    def best_chat_base_url(self) -> Optional[str]:
        """Recommend a chat base_url for the ``endpoint`` provider."""
        if self.llm_endpoint.available and self.llm_endpoint.base_url:
            return self.llm_endpoint.base_url
        return "http://localhost:8080/v1"

    @property
    def best_rerank(self) -> Optional[str]:
        """No first-class rerank backend exists; rerank is moving to LLM-rerank."""
        return None


# =============================================================================
# Network helpers
# =============================================================================


def _get_local_ip() -> Optional[str]:
    """Get the local machine's IP address on the LAN."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None


# =============================================================================
# OpenAI-compatible HTTP-server detection
# =============================================================================

# Common local ports for OpenAI-compatible servers, ordered by preference:
# - 8080: llama.cpp's llama-server default
# - 8000: vLLM, LM Studio default
# - 1234: LM Studio's older default
# - 11434: Ollama (only if ``/v1/`` mode is enabled)
_LOCAL_OPENAI_PORTS: tuple[int, ...] = (8080, 8000, 1234, 11434)
_PROBE_TIMEOUT_SECONDS = 0.5


def detect_llm_endpoint() -> ServiceStatus:
    """
    Detect a local OpenAI-compatible HTTP server.

    Probes common ports for ``GET /v1/models`` (the canonical OpenAI
    endpoint that every compliant server implements). Returns the first
    responsive server.
    """
    try:
        import httpx
    except ImportError:
        return ServiceStatus(
            name="OpenAI-compatible endpoint",
            available=False,
            details="httpx not installed",
        )

    for port in _LOCAL_OPENAI_PORTS:
        for host in ("localhost", "127.0.0.1"):
            base_url = f"http://{host}:{port}/v1"
            try:
                response = httpx.get(
                    f"{base_url}/models",
                    timeout=_PROBE_TIMEOUT_SECONDS,
                )
            except Exception as e:
                logger.debug(f"No OpenAI-compatible server at {base_url}: {e}")
                continue

            if response.status_code != 200:
                continue

            try:
                data = response.json()
            except Exception:
                continue

            models = data.get("data") or data.get("models") or []
            model_ids = [m.get("id") or m.get("name") or "?" for m in models[:3]]
            if model_ids:
                details = f"Models: {', '.join(str(m) for m in model_ids)}"
                if len(models) > 3:
                    details += f" (+{len(models) - 3} more)"
            else:
                details = "Server reachable; no models listed"

            return ServiceStatus(
                name="OpenAI-compatible endpoint",
                available=True,
                host=host,
                port=port,
                base_url=base_url,
                details=details,
            )

    return ServiceStatus(
        name="OpenAI-compatible endpoint",
        available=False,
        details=(
            "Not running. Start a server (e.g. `llama-server -m model.gguf "
            "--port 8080`) or set OPENAI_API_KEY for cloud."
        ),
    )


# =============================================================================
# API key detection
# =============================================================================

# The only first-class cloud preset is ``openai``. Other clouds are
# reachable via the ``endpoint`` provider with a custom api_key_env.
_API_KEY_ENV_VARS: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
}


def detect_api_key(provider: str) -> ApiKeyStatus:
    """
    Check if an API key environment variable is set.

    Args:
        provider: One of the supported preset names (currently only
            ``openai``).
    """
    provider_lower = provider.lower()
    env_var = _API_KEY_ENV_VARS.get(provider_lower)

    if not env_var:
        return ApiKeyStatus(
            name=provider,
            available=False,
            env_var="",
            details=(
                f"Unknown preset: {provider}. Use the 'endpoint' "
                f"provider with auth.api_key_env for arbitrary clouds."
            ),
        )

    key = os.getenv(env_var)
    if key:
        masked = f"{key[:8]}..." if len(key) > 8 else key
        return ApiKeyStatus(
            name=provider.capitalize(),
            available=True,
            env_var=env_var,
            details=f"Set ({masked})",
        )
    return ApiKeyStatus(
        name=provider.capitalize(),
        available=False,
        env_var=env_var,
        details=f"Not set (export {env_var}=...)",
    )


def detect_system_status() -> SystemStatus:
    """Get a snapshot of detectable system services."""
    return SystemStatus(
        llm_endpoint=detect_llm_endpoint(),
        api_keys={
            "openai": detect_api_key("openai"),
        },
    )

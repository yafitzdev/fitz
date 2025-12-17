# fitz/core/detect.py
"""
Centralized service detection for Fitz.

This module provides auto-discovery of external services (Qdrant, Ollama, etc.)
and is used by:
- CLI commands (doctor, init, quickstart)
- Vector DB plugins (auto-detection of Qdrant host)
- LLM plugins (auto-detection of Ollama)

Usage:
    from fitz.core.detect import detect_qdrant, detect_ollama, detect_api_key

    # Check Qdrant
    qdrant = detect_qdrant()
    if qdrant.available:
        print(f"Qdrant at {qdrant.host}:{qdrant.port}")

    # Check API keys
    cohere = detect_api_key("cohere")
    if cohere.available:
        print(f"Cohere key set: {cohere.details}")
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class ServiceStatus:
    """Status of a detected service."""

    name: str
    available: bool
    host: Optional[str] = None
    port: Optional[int] = None
    details: str = ""


@dataclass
class ApiKeyStatus:
    """Status of an API key."""

    name: str
    available: bool
    env_var: str = ""
    details: str = ""


@dataclass
class SystemStatus:
    """Complete system status."""

    ollama: ServiceStatus
    qdrant: ServiceStatus
    faiss: ServiceStatus
    api_keys: dict[str, ApiKeyStatus]

    @property
    def best_llm(self) -> str:
        """Return the best available LLM provider."""
        if self.api_keys["cohere"].available:
            return "cohere"
        if self.api_keys["openai"].available:
            return "openai"
        if self.api_keys["anthropic"].available:
            return "anthropic"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # Default fallback

    @property
    def best_embedding(self) -> str:
        """Return the best available embedding provider."""
        if self.api_keys["cohere"].available:
            return "cohere"
        if self.api_keys["openai"].available:
            return "openai"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # Default fallback

    @property
    def best_vector_db(self) -> str:
        """Return the best available vector database."""
        if self.qdrant.available:
            return "qdrant"
        if self.faiss.available:
            return "faiss"
        return "faiss"  # Default fallback


# =============================================================================
# Service Detection
# =============================================================================


def detect_ollama() -> ServiceStatus:
    """
    Detect if Ollama is running.

    Checks HTTP endpoint at localhost:11434 and 127.0.0.1:11434.

    Returns:
        ServiceStatus with available=True if Ollama responds
    """
    try:
        import httpx
    except ImportError:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details="httpx not installed",
        )

    hosts_to_try = ["localhost", "127.0.0.1"]
    port = 11434

    for host in hosts_to_try:
        try:
            response = httpx.get(
                f"http://{host}:{port}/api/tags",
                timeout=2.0,
            )
            if response.status_code == 200:
                data = response.json()
                models = data.get("models", [])
                model_names = [m.get("name", "?") for m in models[:3]]

                if model_names:
                    details = f"Models: {', '.join(model_names)}"
                    if len(models) > 3:
                        details += f" (+{len(models) - 3} more)"
                else:
                    details = "No models installed"

                return ServiceStatus(
                    name="Ollama",
                    available=True,
                    host=host,
                    port=port,
                    details=details,
                )
        except Exception as e:
            logger.debug(f"Ollama not found at {host}:{port}: {e}")
            continue

    return ServiceStatus(
        name="Ollama",
        available=False,
        details="Not running (tried localhost:11434)",
    )


def detect_qdrant() -> ServiceStatus:
    """
    Detect if Qdrant is accessible.

    Checks these addresses in order:
    1. QDRANT_HOST env var (if set)
    2. localhost
    3. 127.0.0.1

    Returns:
        ServiceStatus with host/port if found
    """
    try:
        import httpx
    except ImportError:
        return ServiceStatus(
            name="Qdrant",
            available=False,
            details="httpx not installed",
        )

    # Build list of hosts to try
    env_host = os.getenv("QDRANT_HOST")
    port = int(os.getenv("QDRANT_PORT", "6333"))

    hosts_to_try = []
    if env_host:
        hosts_to_try.append(env_host)
    hosts_to_try.extend(["localhost", "127.0.0.1"])

    for host in hosts_to_try:
        try:
            response = httpx.get(
                f"http://{host}:{port}/collections",
                timeout=2.0,
            )
            if response.status_code == 200:
                data = response.json()
                collections = data.get("result", {}).get("collections", [])
                col_names = [c.get("name", "?") for c in collections[:3]]

                if col_names:
                    details = f"Collections: {', '.join(col_names)}"
                    if len(collections) > 3:
                        details += f" (+{len(collections) - 3} more)"
                else:
                    details = "No collections"

                return ServiceStatus(
                    name="Qdrant",
                    available=True,
                    host=host,
                    port=port,
                    details=details,
                )
        except Exception as e:
            logger.debug(f"Qdrant not found at {host}:{port}: {e}")
            continue

    return ServiceStatus(
        name="Qdrant",
        available=False,
        details=f"Not running (tried localhost:{port})",
    )


def detect_faiss() -> ServiceStatus:
    """
    Check if FAISS is installed.

    Returns:
        ServiceStatus with available=True if faiss can be imported
    """
    try:
        import faiss

        return ServiceStatus(
            name="FAISS",
            available=True,
            details="Installed (local vector DB)",
        )
    except ImportError:
        return ServiceStatus(
            name="FAISS",
            available=False,
            details="Not installed (pip install faiss-cpu)",
        )


def detect_api_key(provider: str) -> ApiKeyStatus:
    """
    Check if an API key is set for a provider.

    Args:
        provider: Provider name (cohere, openai, anthropic)

    Returns:
        ApiKeyStatus with available=True if key is set
    """
    env_var_map = {
        "cohere": "COHERE_API_KEY",
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
    }

    provider_lower = provider.lower()
    env_var = env_var_map.get(provider_lower)

    if not env_var:
        return ApiKeyStatus(
            name=provider,
            available=False,
            env_var="",
            details=f"Unknown provider: {provider}",
        )

    key = os.getenv(env_var)

    if key:
        # Show first 8 chars for verification
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
    """Get complete system status."""
    return SystemStatus(
        ollama=detect_ollama(),
        qdrant=detect_qdrant(),
        faiss=detect_faiss(),
        api_keys={
            "cohere": detect_api_key("cohere"),
            "openai": detect_api_key("openai"),
            "anthropic": detect_api_key("anthropic"),
        },
    )


# =============================================================================
# Connection Helpers
# =============================================================================


def get_qdrant_connection() -> tuple[str, int]:
    """
    Get Qdrant connection details, auto-detecting if needed.

    Returns:
        (host, port) tuple
    """
    status = detect_qdrant()

    if status.available and status.host and status.port:
        return status.host, status.port

    # Fall back to env vars or defaults
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    return host, port


def get_ollama_connection() -> tuple[str, int]:
    """
    Get Ollama connection details, auto-detecting if needed.

    Returns:
        (host, port) tuple
    """
    status = detect_ollama()

    if status.available and status.host and status.port:
        return status.host, status.port

    return "localhost", 11434


# =============================================================================
# Aliases for CLI compatibility
# =============================================================================

# CLI modules use these names
ProviderStatus = ApiKeyStatus  # Alias for backwards compatibility
detect_all = detect_system_status  # Alias for backwards compatibility

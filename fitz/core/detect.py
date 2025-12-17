# fitz/core/detect.py
"""
Centralized service detection for Fitz.

This module provides auto-discovery of external services (Qdrant, Ollama, etc.)
and is used by:
- CLI commands (doctor, init, quickstart)
- Vector DB plugins (QdrantVectorDB)
- LLM plugins (OllamaChatClient, OllamaEmbeddingClient)

The goal is to provide a "just works" experience where Fitz automatically
finds services running on common addresses without manual configuration.

Usage:
    from fitz.core.detect import detect_qdrant, detect_ollama

    # Get Qdrant connection info
    qdrant = detect_qdrant()
    if qdrant.available:
        client = QdrantClient(host=qdrant.host, port=qdrant.port)

    # Get Ollama connection info
    ollama = detect_ollama()
    if ollama.available:
        # Use ollama.host and ollama.port
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class ServiceStatus:
    """
    Status of a detected service.

    Attributes:
        name: Human-readable service name
        available: Whether the service is reachable
        host: Detected host address (if available)
        port: Detected port (if available)
        details: Human-readable status message
        env_var: Environment variable that can override detection
    """

    name: str
    available: bool
    host: Optional[str] = None
    port: Optional[int] = None
    details: str = ""
    env_var: str = ""


@dataclass
class ApiKeyStatus:
    """Status of an API key."""

    name: str
    available: bool
    details: str = ""
    env_var: str = ""


# =============================================================================
# Configuration: Common Addresses to Try
# =============================================================================

# Common Qdrant host addresses (in order of priority)
QDRANT_COMMON_HOSTS: List[str] = [
    "localhost",
    "127.0.0.1",
    "192.168.178.2",  # Common Docker bridge network
    "192.168.1.1",  # Common router/server address
    "host.docker.internal",  # Docker Desktop (Windows/Mac)
    "172.17.0.1",  # Default Docker bridge gateway
]

# Common Ollama host addresses (in order of priority)
OLLAMA_COMMON_HOSTS: List[str] = [
    "localhost",
    "127.0.0.1",
    "0.0.0.0",
    "host.docker.internal",
]

# Default ports
QDRANT_DEFAULT_PORT = 6333
OLLAMA_DEFAULT_PORT = 11434


# =============================================================================
# Qdrant Detection
# =============================================================================


def detect_qdrant(
        timeout: float = 2.0,
        extra_hosts: Optional[List[str]] = None,
) -> ServiceStatus:
    """
    Auto-detect Qdrant server.

    Tries multiple common addresses to find a running Qdrant instance.
    Priority:
    1. QDRANT_HOST environment variable (if set)
    2. Common localhost addresses
    3. Common Docker/network addresses

    Args:
        timeout: Connection timeout in seconds
        extra_hosts: Additional hosts to try (added to the list)

    Returns:
        ServiceStatus with connection details if found

    Example:
        qdrant = detect_qdrant()
        if qdrant.available:
            from qdrant_client import QdrantClient
            client = QdrantClient(host=qdrant.host, port=qdrant.port)
    """
    # Build list of hosts to try - prioritize env var
    env_host = os.getenv("QDRANT_HOST")
    hosts_to_try: List[str] = []

    if env_host:
        hosts_to_try.append(env_host)

    # Add common locations (avoid duplicates)
    for h in QDRANT_COMMON_HOSTS:
        if h not in hosts_to_try:
            hosts_to_try.append(h)

    # Add any extra hosts
    if extra_hosts:
        for h in extra_hosts:
            if h not in hosts_to_try:
                hosts_to_try.append(h)

    port = int(os.getenv("QDRANT_PORT", str(QDRANT_DEFAULT_PORT)))

    for host in hosts_to_try:
        try:
            url = f"http://{host}:{port}/collections"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    import json
                    data = json.loads(response.read())
                    collections = data.get("result", {}).get("collections", [])
                    col_count = len(collections)

                    logger.debug(f"Found Qdrant at {host}:{port} ({col_count} collections)")

                    return ServiceStatus(
                        name="Qdrant",
                        available=True,
                        host=host,
                        port=port,
                        details=f"At {host}:{port} ({col_count} collections)",
                        env_var="QDRANT_HOST",
                    )
        except urllib.error.URLError:
            continue
        except Exception as e:
            logger.debug(f"Qdrant check failed for {host}:{port}: {e}")
            continue

    # Not found - build helpful message
    tried = ", ".join(hosts_to_try[:3])
    if len(hosts_to_try) > 3:
        tried += f" (+{len(hosts_to_try) - 3} more)"

    return ServiceStatus(
        name="Qdrant",
        available=False,
        details=f"Not reachable (tried {tried}). Set QDRANT_HOST env var.",
        env_var="QDRANT_HOST",
    )


def get_qdrant_connection() -> tuple[str, int]:
    """
    Get Qdrant connection parameters with auto-detection.

    This is the main function that plugins should use to get connection info.
    It returns (host, port) with auto-detection if env vars are not set.

    Returns:
        Tuple of (host, port)

    Raises:
        ConnectionError: If Qdrant is not reachable

    Example:
        host, port = get_qdrant_connection()
        client = QdrantClient(host=host, port=port)
    """
    # If both env vars are explicitly set, use them without probing
    env_host = os.getenv("QDRANT_HOST")
    env_port = os.getenv("QDRANT_PORT")

    if env_host and env_port:
        return env_host, int(env_port)

    # Auto-detect
    status = detect_qdrant()

    if status.available:
        return status.host, status.port

    # Fall back to defaults (let the actual connection fail with a clear error)
    return os.getenv("QDRANT_HOST", "localhost"), int(os.getenv("QDRANT_PORT", "6333"))


# =============================================================================
# Ollama Detection
# =============================================================================


def detect_ollama(
        timeout: float = 3.0,
        extra_hosts: Optional[List[str]] = None,
) -> ServiceStatus:
    """
    Auto-detect Ollama server.

    Checks HTTP API first (most reliable), then falls back to CLI check
    to distinguish "not installed" from "installed but not running".

    Args:
        timeout: Connection timeout in seconds
        extra_hosts: Additional hosts to try

    Returns:
        ServiceStatus with connection details if found
    """
    # Get host/port from env or defaults
    env_host = os.getenv("OLLAMA_HOST", "localhost")
    port = int(os.getenv("OLLAMA_PORT", str(OLLAMA_DEFAULT_PORT)))

    # Build hosts to try
    hosts_to_try = [env_host]
    for h in OLLAMA_COMMON_HOSTS:
        if h not in hosts_to_try:
            hosts_to_try.append(h)

    if extra_hosts:
        for h in extra_hosts:
            if h not in hosts_to_try:
                hosts_to_try.append(h)

    # Check HTTP API (PRIMARY - most reliable)
    for host in hosts_to_try:
        try:
            url = f"http://{host}:{port}/api/tags"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    import json
                    data = json.loads(response.read())
                    models = data.get("models", [])
                    model_names = [m.get("name", "?") for m in models]

                    if model_names:
                        model_str = ", ".join(model_names[:3])
                        if len(model_names) > 3:
                            model_str += f" (+{len(model_names) - 3} more)"
                        details = f"Running at {host}:{port} ({model_str})"
                    else:
                        details = f"Running at {host}:{port} (no models)"

                    logger.debug(f"Found Ollama at {host}:{port}")

                    return ServiceStatus(
                        name="Ollama",
                        available=True,
                        host=host,
                        port=port,
                        details=details,
                        env_var="OLLAMA_HOST",
                    )
        except urllib.error.URLError:
            continue
        except Exception as e:
            logger.debug(f"Ollama check failed for {host}:{port}: {e}")
            continue

    # Check CLI (to distinguish "not installed" from "not running")
    cli_installed = _check_ollama_cli()

    if cli_installed:
        return ServiceStatus(
            name="Ollama",
            available=False,
            details=f"Installed but not running at {env_host}:{port} (run: ollama serve)",
            env_var="OLLAMA_HOST",
        )

    return ServiceStatus(
        name="Ollama",
        available=False,
        details="Not installed (https://ollama.com)",
        env_var="OLLAMA_HOST",
    )


def _check_ollama_cli() -> bool:
    """Check if Ollama CLI is installed (without checking if service is running)."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            shell=(sys.platform == "win32"),  # Use shell on Windows
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    except Exception:
        return False


def get_ollama_connection() -> tuple[str, int]:
    """
    Get Ollama connection parameters with auto-detection.

    Returns:
        Tuple of (host, port)
    """
    env_host = os.getenv("OLLAMA_HOST")
    env_port = os.getenv("OLLAMA_PORT")

    if env_host and env_port:
        return env_host, int(env_port)

    status = detect_ollama()

    if status.available:
        return status.host, status.port

    return os.getenv("OLLAMA_HOST", "localhost"), int(os.getenv("OLLAMA_PORT", "11434"))


# =============================================================================
# FAISS Detection (local, no network)
# =============================================================================


def detect_faiss() -> ServiceStatus:
    """Check if FAISS is installed."""
    try:
        import faiss  # noqa: F401
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


# =============================================================================
# API Key Detection
# =============================================================================


def detect_api_key(name: str, env_var: str) -> ApiKeyStatus:
    """Check if an API key is set in environment."""
    key = os.getenv(env_var)
    if key:
        # Mask the key for display
        masked = f"{key[:8]}..." if len(key) > 8 else "***"
        return ApiKeyStatus(
            name=name,
            available=True,
            details=f"Set ({masked})",
            env_var=env_var,
        )
    return ApiKeyStatus(
        name=name,
        available=False,
        details=f"Not set (export {env_var}=...)",
        env_var=env_var,
    )


def detect_api_keys() -> dict[str, ApiKeyStatus]:
    """Detect all supported API keys."""
    return {
        "cohere": detect_api_key("Cohere", "COHERE_API_KEY"),
        "openai": detect_api_key("OpenAI", "OPENAI_API_KEY"),
        "anthropic": detect_api_key("Anthropic", "ANTHROPIC_API_KEY"),
    }


# =============================================================================
# Combined Detection
# =============================================================================


@dataclass
class SystemStatus:
    """Complete system status from all detection checks."""

    api_keys: dict[str, ApiKeyStatus]
    ollama: ServiceStatus
    qdrant: ServiceStatus
    faiss: ServiceStatus

    @property
    def has_llm(self) -> bool:
        """Check if any LLM provider is available."""
        return (
                self.ollama.available
                or self.api_keys.get("cohere", ApiKeyStatus("", False)).available
                or self.api_keys.get("openai", ApiKeyStatus("", False)).available
                or self.api_keys.get("anthropic", ApiKeyStatus("", False)).available
        )

    @property
    def has_embedding(self) -> bool:
        """Check if any embedding provider is available."""
        return (
                self.ollama.available
                or self.api_keys.get("cohere", ApiKeyStatus("", False)).available
                or self.api_keys.get("openai", ApiKeyStatus("", False)).available
        )

    @property
    def has_vector_db(self) -> bool:
        """Check if any vector DB is available."""
        return self.qdrant.available or self.faiss.available

    @property
    def best_llm(self) -> str:
        """Get the best available LLM provider name."""
        if self.api_keys.get("cohere", ApiKeyStatus("", False)).available:
            return "cohere"
        if self.api_keys.get("openai", ApiKeyStatus("", False)).available:
            return "openai"
        if self.api_keys.get("anthropic", ApiKeyStatus("", False)).available:
            return "anthropic"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # fallback

    @property
    def best_embedding(self) -> str:
        """Get the best available embedding provider name."""
        if self.api_keys.get("cohere", ApiKeyStatus("", False)).available:
            return "cohere"
        if self.api_keys.get("openai", ApiKeyStatus("", False)).available:
            return "openai"
        if self.ollama.available:
            return "ollama"
        return "ollama"  # fallback

    @property
    def best_vector_db(self) -> str:
        """Get the best available vector DB name."""
        if self.qdrant.available:
            return "qdrant"
        if self.faiss.available:
            return "faiss"
        return "faiss"  # fallback

    @property
    def qdrant_host(self) -> str:
        """Get the detected Qdrant host."""
        return self.qdrant.host or "localhost"

    @property
    def qdrant_port(self) -> int:
        """Get the detected Qdrant port."""
        return self.qdrant.port or QDRANT_DEFAULT_PORT


def detect_all() -> SystemStatus:
    """
    Run all detection checks and return complete system status.

    This is the main entry point for comprehensive system detection.

    Example:
        from fitz.core.detect import detect_all

        system = detect_all()
        if system.has_llm and system.has_vector_db:
            print("Ready to run RAG pipeline!")
            print(f"Using {system.best_llm} for LLM")
            print(f"Using Qdrant at {system.qdrant_host}:{system.qdrant_port}")
    """
    return SystemStatus(
        api_keys=detect_api_keys(),
        ollama=detect_ollama(),
        qdrant=detect_qdrant(),
        faiss=detect_faiss(),
    )
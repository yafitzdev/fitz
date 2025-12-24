# fitz_ai/core/detect.py
"""
Centralized service detection for Fitz.

This module provides auto-discovery of external services (Qdrant, Ollama, etc.)
and is used by:
- CLI commands (doctor, init, quickstart)
- Vector DB plugins (auto-detection of Qdrant host)
- LLM plugins (auto-detection of Ollama)

Usage:
    from fitz_ai.core.detect import detect_qdrant, detect_ollama, detect_api_key

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
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

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

    @property
    def best_rerank(self) -> Optional[str]:
        """Return the best available rerank provider (or None if none available)."""
        # Currently only Cohere supports reranking
        if self.api_keys["cohere"].available:
            return "cohere"
        return None

    @property
    def qdrant_host(self) -> str:
        """Return the detected Qdrant host or default."""
        if self.qdrant.available and self.qdrant.host:
            return self.qdrant.host
        return os.getenv("QDRANT_HOST", "localhost")

    @property
    def qdrant_port(self) -> int:
        """Return the detected Qdrant port or default."""
        if self.qdrant.available and self.qdrant.port:
            return self.qdrant.port
        return int(os.getenv("QDRANT_PORT", "6333"))


# =============================================================================
# Network Helpers
# =============================================================================


def _get_local_ip() -> Optional[str]:
    """Get the local machine's IP address on the LAN."""
    try:
        # Create a socket to determine local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0.1)
        # Connect to a public IP (doesn't actually send data)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return None


def _get_all_local_ips() -> List[str]:
    """
    Get all local IP addresses from all network interfaces.

    This handles machines with multiple NICs (e.g., WiFi + Ethernet,
    or machines connected to multiple networks).
    """
    ips = []

    try:
        # Method 1: Use socket to get hostname-based IPs
        hostname = socket.gethostname()
        try:
            # getaddrinfo returns all IPs for the hostname
            for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
                ip = info[4][0]
                if ip not in ips and not ip.startswith("127."):
                    ips.append(ip)
        except socket.gaierror:
            pass

        # Method 2: Try the UDP trick for the "primary" interface
        primary_ip = _get_local_ip()
        if primary_ip and primary_ip not in ips:
            ips.append(primary_ip)

    except Exception as e:
        logger.debug(f"Error getting local IPs: {e}")

    return ips


def _get_subnets_from_ips(ips: List[str]) -> List[str]:
    """Extract unique /24 subnets from a list of IPs."""
    subnets = []
    for ip in ips:
        parts = ip.split(".")
        if len(parts) == 4:
            subnet = ".".join(parts[:3])
            if subnet not in subnets:
                subnets.append(subnet)
    return subnets


def _check_host_port(host: str, port: int, timeout: float = 0.5) -> bool:
    """
    Quick TCP check if a host:port is reachable.

    This is faster than a full HTTP request for initial filtering.
    """
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        result = sock.connect_ex((host, port))
        sock.close()
        return result == 0
    except Exception:
        return False


def _check_qdrant_http(host: str, port: int, timeout: float = 1.0) -> Optional[dict]:
    """
    Check if Qdrant is running at host:port via HTTP.

    Returns the collections response if successful, None otherwise.
    """
    try:
        import httpx

        response = httpx.get(
            f"http://{host}:{port}/collections",
            timeout=timeout,
        )
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return None


def _get_config_qdrant_host() -> Optional[str]:
    """
    Try to get Qdrant host from existing Fitz config.

    This allows detection to use a previously configured host.
    """
    try:
        from fitz_ai.core.config import load_config_dict
        from fitz_ai.core.paths import FitzPaths

        config_path = FitzPaths.config()
        if config_path.exists():
            config = load_config_dict(config_path)
            # Check for qdrant host in vector_db config
            vector_db = config.get("vector_db", {})
            if vector_db.get("plugin_name") == "qdrant":
                host = vector_db.get("host")
                if host:
                    return host
    except Exception:
        pass
    return None


def _build_lan_scan_hosts(port: int) -> List[str]:
    """
    Build a comprehensive list of hosts to scan for LAN services.

    Returns hosts in priority order:
    1. QDRANT_HOST env var (if set)
    2. Host from existing Fitz config (if available)
    3. localhost / 127.0.0.1
    4. All detected local subnets with common server addresses

    This handles:
    - Multi-NIC machines (scans all detected subnets)
    - Common server IP patterns (.1, .2, .10, .100, .200, .254)
    - Docker default bridge (172.17.0.x)
    """
    hosts = []
    seen = set()

    def add_host(h: str):
        if h not in seen:
            seen.add(h)
            hosts.append(h)

    # 1. Environment variable (highest priority)
    env_host = os.getenv("QDRANT_HOST")
    if env_host:
        add_host(env_host)

    # 2. Previously configured host
    config_host = _get_config_qdrant_host()
    if config_host:
        add_host(config_host)

    # 3. Localhost variants
    add_host("localhost")
    add_host("127.0.0.1")

    # 4. Get all local IPs and their subnets
    local_ips = _get_all_local_ips()
    subnets = _get_subnets_from_ips(local_ips)

    # Common IP endings where servers/services typically run
    # Expanded from original [1, 2, 100, 254] to catch more cases
    common_endings = [
        1,  # Gateway/router
        2,  # Often first server after gateway
        3,
        4,
        5,  # Small network servers
        10,  # Common static IP
        50,  # Mid-range static
        100,  # DHCP range start or static
        150,  # Mid-range
        200,  # Common static
        254,  # Last usable (often used for servers)
    ]

    for subnet in subnets:
        for ending in common_endings:
            add_host(f"{subnet}.{ending}")

    # 5. Docker default bridge network (172.17.0.x)
    # Common for Docker-hosted Qdrant
    docker_subnet = "172.17.0"
    for ending in [1, 2, 3, 4, 5]:
        add_host(f"{docker_subnet}.{ending}")

    return hosts


def _scan_hosts_concurrent(
    hosts: List[str],
    port: int,
    check_fn,
    max_workers: int = 10,
    timeout: float = 0.5,
) -> Optional[Tuple[str, any]]:
    """
    Scan multiple hosts concurrently and return the first successful result.

    Args:
        hosts: List of hosts to scan
        port: Port to check
        check_fn: Function(host, port, timeout) -> result or None
        max_workers: Max concurrent connections
        timeout: Per-host timeout

    Returns:
        Tuple of (host, result) for first successful host, or None
    """
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all checks
        future_to_host = {
            executor.submit(check_fn, host, port, timeout): host for host in hosts
        }

        # Return first successful result
        for future in as_completed(future_to_host):
            host = future_to_host[future]
            try:
                result = future.result()
                if result is not None:
                    # Cancel remaining futures
                    for f in future_to_host:
                        f.cancel()
                    return (host, result)
            except Exception:
                continue

    return None


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

    Uses concurrent scanning for fast LAN detection. Checks:
    1. QDRANT_HOST env var (if set)
    2. Previously configured host from .fitz/config.yaml
    3. localhost / 127.0.0.1
    4. Common LAN addresses on all detected subnets
    5. Docker bridge network (172.17.0.x)

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

    port = int(os.getenv("QDRANT_PORT", "6333"))

    # Build comprehensive host list
    hosts_to_try = _build_lan_scan_hosts(port)

    logger.debug(f"Scanning {len(hosts_to_try)} hosts for Qdrant on port {port}")

    # Use concurrent scanning for speed
    result = _scan_hosts_concurrent(
        hosts=hosts_to_try,
        port=port,
        check_fn=_check_qdrant_http,
        max_workers=15,  # Scan up to 15 hosts in parallel
        timeout=1.0,
    )

    if result:
        host, data = result
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

    # Build informative error message
    local_ips = _get_all_local_ips()
    subnets = _get_subnets_from_ips(local_ips)

    if subnets:
        subnet_str = ", ".join(f"{s}.x" for s in subnets[:2])
        if len(subnets) > 2:
            subnet_str += f" +{len(subnets) - 2} more"
        details = f"Not found (scanned localhost and {subnet_str} on port {port})"
    else:
        details = f"Not found (scanned localhost:{port}, couldn't detect LAN)"

    return ServiceStatus(
        name="Qdrant",
        available=False,
        details=details,
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

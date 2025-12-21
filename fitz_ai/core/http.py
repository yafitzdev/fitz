# fitz_ai/core/http.py
"""
Centralized HTTP client factory for API integrations.

This module provides a consistent way to create HTTP clients across all
plugins, with standardized error handling, timeouts, and headers.

Usage:
    from fitz_ai.core.http import create_api_client, APIError, handle_api_error

    # Create a client for an API
    client = create_api_client(
        base_url="https://api.cohere.ai/v1",
        api_key="your-key",
        timeout=30.0,
    )

    # Make requests
    response = client.post("/embed", json=payload)

    # Or use the context manager
    with create_api_client(...) as client:
        response = client.post("/chat", json=payload)

Design principles:
    - Single place to configure HTTP behavior
    - Consistent error handling across all plugins
    - Easy to add retries, logging, etc. in one place
    - Supports both sync and async clients
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Generator, Optional

logger = logging.getLogger(__name__)

# Check for httpx availability
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    httpx = None  # type: ignore
    HTTPX_AVAILABLE = False


# =============================================================================
# Errors
# =============================================================================


class HTTPClientError(Exception):
    """Base error for HTTP client issues."""

    pass


class HTTPClientNotAvailable(HTTPClientError):
    """Raised when httpx is not installed."""

    def __init__(self):
        super().__init__(
            "httpx is required for HTTP API calls. Install with: pip install httpx"
        )


@dataclass
class APIError(Exception):
    """
    Structured API error with details.

    Attributes:
        message: Human-readable error message
        status_code: HTTP status code (if available)
        provider: API provider name (e.g., "cohere", "openai")
        endpoint: API endpoint that failed
        details: Additional error details from the API response
        original_error: The original exception that caused this error
    """

    message: str
    status_code: Optional[int] = None
    provider: Optional[str] = None
    endpoint: Optional[str] = None
    details: Optional[str] = None
    original_error: Optional[Exception] = None

    def __str__(self) -> str:
        parts = [self.message]

        if self.status_code:
            parts.append(f"(HTTP {self.status_code})")

        if self.details:
            parts.append(f"- {self.details}")

        return " ".join(parts)


class RateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    pass


class AuthenticationError(APIError):
    """Raised when API authentication fails."""

    pass


class ModelNotFoundError(APIError):
    """Raised when requested model doesn't exist."""

    pass


# =============================================================================
# Default Configuration
# =============================================================================

# Default timeouts by use case
DEFAULT_TIMEOUTS = {
    "default": 30.0,
    "chat": 120.0,  # LLM generation can be slow
    "embedding": 30.0,
    "rerank": 30.0,
    "health_check": 5.0,
}

# Default headers
DEFAULT_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


# =============================================================================
# Client Factory
# =============================================================================


def create_api_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    timeout_type: str = "default",
    headers: Optional[Dict[str, str]] = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    **kwargs: Any,
) -> "httpx.Client":
    """
    Create a configured HTTP client for API calls.

    Args:
        base_url: Base URL for the API (e.g., "https://api.cohere.ai/v1")
        api_key: API key for authentication (optional)
        timeout: Request timeout in seconds (or use timeout_type)
        timeout_type: Preset timeout type ("default", "chat", "embedding", etc.)
        headers: Additional headers to include
        auth_header: Header name for authentication (default: "Authorization")
        auth_scheme: Authentication scheme (default: "Bearer")
        **kwargs: Additional arguments passed to httpx.Client

    Returns:
        Configured httpx.Client instance

    Raises:
        HTTPClientNotAvailable: If httpx is not installed

    Example:
        client = create_api_client(
            base_url="https://api.cohere.ai/v1",
            api_key=os.getenv("COHERE_API_KEY"),
            timeout_type="chat",
        )

        response = client.post("/chat", json={"message": "Hello"})
    """
    if not HTTPX_AVAILABLE:
        raise HTTPClientNotAvailable()

    # Resolve timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUTS.get(timeout_type, DEFAULT_TIMEOUTS["default"])

    # Build headers
    final_headers = dict(DEFAULT_HEADERS)

    if api_key:
        final_headers[auth_header] = f"{auth_scheme} {api_key}"

    if headers:
        final_headers.update(headers)

    # Create client
    client = httpx.Client(
        base_url=base_url,
        headers=final_headers,
        timeout=timeout,
        **kwargs,
    )

    logger.debug(f"Created HTTP client for {base_url} (timeout={timeout}s)")

    return client


@contextmanager
def api_client(
    base_url: str,
    api_key: Optional[str] = None,
    **kwargs: Any,
) -> Generator["httpx.Client", None, None]:
    """
    Context manager for HTTP client with automatic cleanup.

    Usage:
        with api_client("https://api.cohere.ai/v1", api_key=key) as client:
            response = client.post("/chat", json=payload)
    """
    client = create_api_client(base_url, api_key, **kwargs)
    try:
        yield client
    finally:
        client.close()


# =============================================================================
# Error Handling
# =============================================================================


def handle_api_error(
    exc: Exception,
    provider: str = "unknown",
    endpoint: str = "",
) -> APIError:
    """
    Convert an httpx exception to a structured APIError.

    Args:
        exc: The original exception
        provider: Name of the API provider
        endpoint: The endpoint that was called

    Returns:
        Appropriate APIError subclass

    Example:
        try:
            response = client.post("/chat", json=payload)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise handle_api_error(exc, provider="cohere", endpoint="/chat")
    """
    if not HTTPX_AVAILABLE:
        return APIError(
            message=str(exc),
            provider=provider,
            endpoint=endpoint,
            original_error=exc,
        )

    # Handle HTTP status errors
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code

        # Try to extract error details from response
        details = None
        try:
            error_data = exc.response.json()
            details = error_data.get("message") or error_data.get("error", {}).get(
                "message"
            )
        except Exception:
            details = exc.response.text[:200] if exc.response.text else None

        # Map to specific error types
        if status_code == 401:
            return AuthenticationError(
                message=f"{provider} authentication failed",
                status_code=status_code,
                provider=provider,
                endpoint=endpoint,
                details=details,
                original_error=exc,
            )
        elif status_code == 429:
            return RateLimitError(
                message=f"{provider} rate limit exceeded",
                status_code=status_code,
                provider=provider,
                endpoint=endpoint,
                details=details,
                original_error=exc,
            )
        elif status_code == 404:
            return ModelNotFoundError(
                message=f"{provider} resource not found",
                status_code=status_code,
                provider=provider,
                endpoint=endpoint,
                details=details,
                original_error=exc,
            )
        else:
            return APIError(
                message=f"{provider} API request failed",
                status_code=status_code,
                provider=provider,
                endpoint=endpoint,
                details=details,
                original_error=exc,
            )

    # Handle connection errors
    if isinstance(exc, httpx.ConnectError):
        return APIError(
            message=f"Failed to connect to {provider}",
            provider=provider,
            endpoint=endpoint,
            details=str(exc),
            original_error=exc,
        )

    # Handle timeout errors
    if isinstance(exc, httpx.TimeoutException):
        return APIError(
            message=f"{provider} request timed out",
            provider=provider,
            endpoint=endpoint,
            details="Consider increasing the timeout for this operation",
            original_error=exc,
        )

    # Generic fallback
    return APIError(
        message=f"{provider} request failed: {exc}",
        provider=provider,
        endpoint=endpoint,
        original_error=exc,
    )


def raise_for_status(
    response: "httpx.Response",
    provider: str = "unknown",
    endpoint: str = "",
) -> None:
    """
    Check response status and raise appropriate APIError if failed.

    Args:
        response: The httpx response to check
        provider: Name of the API provider
        endpoint: The endpoint that was called

    Raises:
        APIError: If the response indicates an error

    Example:
        response = client.post("/chat", json=payload)
        raise_for_status(response, provider="cohere", endpoint="/chat")
        data = response.json()
    """
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise handle_api_error(exc, provider=provider, endpoint=endpoint) from exc


# =============================================================================
# Health Check Utility
# =============================================================================


def check_api_health(
    url: str,
    timeout: float = 5.0,
    expected_status: int = 200,
) -> bool:
    """
    Simple health check for an API endpoint.

    Args:
        url: Full URL to check
        timeout: Request timeout in seconds
        expected_status: Expected HTTP status code

    Returns:
        True if the endpoint is healthy, False otherwise

    Example:
        if check_api_health("https://api.cohere.ai/v1/health"):
            print("Cohere API is up")
    """
    if not HTTPX_AVAILABLE:
        return False

    try:
        response = httpx.get(url, timeout=timeout)
        return response.status_code == expected_status
    except Exception:
        return False


def check_api_auth(
    base_url: str,
    api_key: str,
    test_endpoint: str = "/models",
    auth_scheme: str = "Bearer",
) -> bool:
    """
    Check if API authentication is valid.

    Args:
        base_url: Base URL for the API
        api_key: API key to test
        test_endpoint: Endpoint to test (should be lightweight)
        auth_scheme: Authentication scheme

    Returns:
        True if authentication is valid, False otherwise
    """
    if not HTTPX_AVAILABLE:
        return False

    try:
        with api_client(
            base_url=base_url,
            api_key=api_key,
            timeout=5.0,
            auth_scheme=auth_scheme,
        ) as client:
            response = client.get(test_endpoint)
            return response.status_code != 401
    except Exception:
        return False


# =============================================================================
# Convenience: Pre-configured Client Factories
# =============================================================================


def create_cohere_client(
    api_key: str,
    timeout_type: str = "default",
) -> "httpx.Client":
    """Create a client configured for Cohere API."""
    return create_api_client(
        base_url="https://api.cohere.ai/v1",
        api_key=api_key,
        timeout_type=timeout_type,
    )


def create_openai_client(
    api_key: str,
    timeout_type: str = "default",
    base_url: str = "https://api.openai.com/v1",
) -> "httpx.Client":
    """Create a client configured for OpenAI API (or compatible)."""
    return create_api_client(
        base_url=base_url,
        api_key=api_key,
        timeout_type=timeout_type,
    )


# =============================================================================
# Async Support (Optional)
# =============================================================================


def create_async_api_client(
    base_url: str,
    api_key: Optional[str] = None,
    timeout: Optional[float] = None,
    timeout_type: str = "default",
    headers: Optional[Dict[str, str]] = None,
    auth_header: str = "Authorization",
    auth_scheme: str = "Bearer",
    **kwargs: Any,
) -> "httpx.AsyncClient":
    """
    Create an async HTTP client for API calls.

    Same as create_api_client but returns an AsyncClient.

    Usage:
        async with create_async_api_client(...) as client:
            response = await client.post("/chat", json=payload)
    """
    if not HTTPX_AVAILABLE:
        raise HTTPClientNotAvailable()

    # Resolve timeout
    if timeout is None:
        timeout = DEFAULT_TIMEOUTS.get(timeout_type, DEFAULT_TIMEOUTS["default"])

    # Build headers
    final_headers = dict(DEFAULT_HEADERS)

    if api_key:
        final_headers[auth_header] = f"{auth_scheme} {api_key}"

    if headers:
        final_headers.update(headers)

    return httpx.AsyncClient(
        base_url=base_url,
        headers=final_headers,
        timeout=timeout,
        **kwargs,
    )

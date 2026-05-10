# tests/unit/llm/test_noauth.py
"""
Unit tests for NoAuth — the no-op authentication provider used with
OpenAI-compatible endpoints that don't require authentication
(local llama-server, vLLM, etc.).
"""

from __future__ import annotations

from fitz_sage.llm.auth import AuthProvider, NoAuth


class TestNoAuth:
    """NoAuth produces no headers, no kwargs, satisfies AuthProvider."""

    def test_implements_auth_provider_protocol(self) -> None:
        """NoAuth satisfies the AuthProvider runtime protocol."""
        auth = NoAuth()
        assert isinstance(auth, AuthProvider)

    def test_get_headers_is_empty(self) -> None:
        """No headers means no auth applied to requests."""
        auth = NoAuth()
        assert auth.get_headers() == {}

    def test_get_request_kwargs_is_empty(self) -> None:
        """No extra kwargs (no certs, no special timeouts)."""
        auth = NoAuth()
        assert auth.get_request_kwargs() == {}

    def test_multiple_calls_consistent(self) -> None:
        """Repeat calls return the same empty dicts."""
        auth = NoAuth()
        for _ in range(3):
            assert auth.get_headers() == {}
            assert auth.get_request_kwargs() == {}

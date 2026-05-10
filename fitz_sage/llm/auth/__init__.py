# fitz_sage/llm/auth/__init__.py
"""
Authentication providers for LLM clients.

Supports:
- No authentication (NoAuth) — for local llama-server, vLLM without auth, etc.
- API key authentication (simple, env var based)
- M2M OAuth2 client credentials (enterprise, auto-refresh)
- Dynamic httpx.Auth wrapper for SDK integration (DynamicHttpxAuth)
- Token provider callback for OpenAI SDK (TokenProviderAdapter)
- Composite auth for multi-header scenarios (CompositeAuth)
"""

from fitz_sage.llm.auth.api_key import ApiKeyAuth
from fitz_sage.llm.auth.base import AuthProvider
from fitz_sage.llm.auth.certificates import (
    CertificateError,
    validate_certificate_file,
    validate_key_file,
)
from fitz_sage.llm.auth.composite import CompositeAuth
from fitz_sage.llm.auth.httpx_auth import DynamicHttpxAuth
from fitz_sage.llm.auth.m2m import M2MAuth
from fitz_sage.llm.auth.noauth import NoAuth
from fitz_sage.llm.auth.token_provider import TokenProviderAdapter

__all__ = [
    "AuthProvider",
    "ApiKeyAuth",
    "CertificateError",
    "CompositeAuth",
    "M2MAuth",
    "NoAuth",
    "DynamicHttpxAuth",
    "TokenProviderAdapter",
    "validate_certificate_file",
    "validate_key_file",
]

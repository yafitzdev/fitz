# fitz_sage/llm/auth/noauth.py
"""
No-op authentication provider.

For endpoints that do not require authentication (local llama-server,
vLLM without auth gateway, internal proxies, etc.).
"""

from __future__ import annotations

from typing import Any


class NoAuth:
    """
    No-op authentication provider.

    Used with the ``endpoint`` provider when pointing at an OpenAI-compatible
    HTTP server that does not require an API key — for example, a locally
    running ``llama-server`` or ``vllm`` instance.

    Returns no headers and no extra request kwargs, so HTTP requests are
    made unmodified.
    """

    def get_headers(self) -> dict[str, str]:
        """Return no auth headers."""
        return {}

    def get_request_kwargs(self) -> dict[str, Any]:
        """Return no extra request kwargs."""
        return {}

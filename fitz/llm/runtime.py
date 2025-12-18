# fitz/llm/runtime.py
"""
Generic plugin runtime that executes YAML-defined plugins.

This is the "engine" that reads YAML plugin specs and performs the actual
HTTP calls, credential resolution, and response parsing. Users never write
code - they just define YAML, and this runtime does the work.
"""
from __future__ import annotations

import logging
import os
import re
from typing import Any

from fitz.core.http import (
    APIError,
    HTTPClientNotAvailable,
    create_api_client,
    handle_api_error,
    raise_for_status,
)
from fitz.llm.credentials import CredentialError, resolve_api_key
from fitz.llm.loader import (
    load_chat_plugin,
    load_embedding_plugin,
    load_rerank_plugin,
)
from fitz.llm.schema import (
    AuthType,
    ChatPluginSpec,
    EmbeddingPluginSpec,
    InputWrap,
    RerankPluginSpec,
)
from fitz.llm.transforms import get_transformer

logger = logging.getLogger(__name__)


# =============================================================================
# Response Extraction Utilities
# =============================================================================


def extract_path(data: Any, path: str) -> Any:
    """Extract a value from nested data using dot/bracket notation.

    Examples:
        extract_path({"a": {"b": 1}}, "a.b") -> 1
        extract_path({"items": [{"x": 1}]}, "items[0].x") -> 1
        extract_path({"data": [1, 2, 3]}, "data[0]") -> 1

    Args:
        data: The data structure to extract from
        path: Dot-notation path with optional array indices

    Returns:
        Extracted value

    Raises:
        KeyError: If path doesn't exist
        IndexError: If array index is out of bounds
    """
    if not path:
        return data

    # Split path into parts, handling both dots and brackets
    # "items[0].text" -> ["items", "0", "text"]
    parts = re.split(r'\.|\[|\]', path)
    parts = [p for p in parts if p]  # Remove empty strings

    current = data
    for part in parts:
        if isinstance(current, dict):
            current = current[part]
        elif isinstance(current, (list, tuple)):
            current = current[int(part)]
        else:
            raise KeyError(f"Cannot traverse {type(current).__name__} with key {part!r}")

    return current


# =============================================================================
# Base YAML Plugin Client
# =============================================================================


class YAMLPluginBase:
    """Base class for YAML-driven plugin clients."""

    def __init__(
            self,
            spec: ChatPluginSpec | EmbeddingPluginSpec | RerankPluginSpec,
            **kwargs: Any,
    ) -> None:
        self.spec = spec

        # Merge defaults with user kwargs
        self.params = {**spec.defaults, **kwargs}

        # Resolve any required env vars (e.g., Azure endpoint)
        self._env_values: dict[str, str] = {}
        for req_env in spec.required_env:
            value = kwargs.get(req_env.inject_as) or os.getenv(req_env.name) or req_env.default
            if not value:
                raise ValueError(
                    f"{req_env.name} is required. "
                    f"Set it as an environment variable or pass {req_env.inject_as!r} parameter."
                )
            self._env_values[req_env.inject_as] = value

        # Resolve base URL (may have placeholders)
        self._base_url = self._resolve_placeholders(spec.provider.base_url)

        # Resolve API key if auth is needed
        self._api_key: str | None = None
        if spec.auth.type != AuthType.NONE:
            try:
                self._api_key = resolve_api_key(
                    provider=spec.provider.name,
                    config={"api_key": kwargs.get("api_key")} if kwargs.get("api_key") else None,
                )
            except CredentialError as e:
                raise RuntimeError(str(e)) from e

        # Create HTTP client
        self._client = self._create_client()

    def _resolve_placeholders(self, template: str) -> str:
        """Replace {placeholder} with env values or params."""
        result = template

        # Replace from env values first
        for key, value in self._env_values.items():
            result = result.replace(f"{{{key}}}", value)

        # Replace from params
        for key, value in self.params.items():
            if isinstance(value, str):
                result = result.replace(f"{{{key}}}", value)

        return result

    def _create_client(self) -> Any:
        """Create the HTTP client with appropriate auth headers."""
        extra_headers: dict[str, str] = {}

        if self._api_key and self.spec.auth.type != AuthType.NONE:
            header_value = self.spec.auth.header_format.format(key=self._api_key)
            extra_headers[self.spec.auth.header_name] = header_value

        try:
            return create_api_client(
                base_url=self._base_url,
                api_key=None,  # We handle auth via headers ourselves
                timeout_type=self.spec.plugin_type,
                headers=extra_headers if extra_headers else None,
            )
        except HTTPClientNotAvailable:
            raise RuntimeError(
                "httpx is required for YAML plugins. "
                "Install with: pip install httpx"
            )

    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Make HTTP request to the provider."""
        try:
            response = self._client.request(
                method=self.spec.endpoint.method,
                url=self.spec.endpoint.path,
                json=payload,
            )
            raise_for_status(
                response,
                provider=self.spec.provider.name,
                endpoint=self.spec.endpoint.path,
            )
            return response.json()

        except APIError as exc:
            raise RuntimeError(str(exc)) from exc

        except Exception as exc:
            error = handle_api_error(
                exc,
                provider=self.spec.provider.name,
                endpoint=self.spec.endpoint.path,
            )
            raise RuntimeError(str(error)) from exc

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass


# =============================================================================
# YAML Chat Plugin
# =============================================================================


class YAMLChatClient(YAMLPluginBase):
    """
    Generic chat plugin driven by YAML specification.

    Usage:
        client = YAMLChatClient.from_name("cohere", temperature=0.5)
        response = client.chat([{"role": "user", "content": "Hello!"}])
    """

    plugin_type = "chat"

    def __init__(self, spec: ChatPluginSpec, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self._transformer = get_transformer(spec.request.messages_transform.value)

    @classmethod
    def from_name(cls, plugin_name: str, **kwargs: Any) -> "YAMLChatClient":
        """Create a chat client from plugin name.

        Args:
            plugin_name: Name of the YAML plugin (e.g., "cohere", "openai")
            **kwargs: Override default parameters

        Returns:
            Configured YAMLChatClient
        """
        spec = load_chat_plugin(plugin_name)
        return cls(spec, **kwargs)

    @property
    def plugin_name(self) -> str:
        return self.spec.plugin_name

    def chat(self, messages: list[dict[str, Any]]) -> str:
        """Send chat completion request.

        Args:
            messages: List of message dicts with 'role' and 'content'

        Returns:
            The assistant's response text
        """
        # Transform messages to provider format
        payload = self._transformer.transform(messages)

        # Add static fields
        payload.update(self.spec.request.static_fields)

        # Add mapped parameters
        for fitz_name, provider_name in self.spec.request.param_map.items():
            if fitz_name in self.params:
                # Handle nested param paths like "options.temperature"
                self._set_nested_param(payload, provider_name, self.params[fitz_name])

        # Make request
        response = self._make_request(payload)

        # Extract response content
        try:
            content = extract_path(response, self.spec.response.content_path)
            return str(content) if content else ""
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to extract response: {e}. Full response: {response}")
            return ""

    def _set_nested_param(self, payload: dict, path: str, value: Any) -> None:
        """Set a parameter at a potentially nested path (e.g., 'options.temperature')."""
        parts = path.split(".")
        current = payload
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value


# =============================================================================
# YAML Embedding Plugin
# =============================================================================


class YAMLEmbeddingClient(YAMLPluginBase):
    """
    Generic embedding plugin driven by YAML specification.

    Usage:
        client = YAMLEmbeddingClient.from_name("cohere")
        embedding = client.embed("Hello, world!")
    """

    plugin_type = "embedding"

    def __init__(self, spec: EmbeddingPluginSpec, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)

    @classmethod
    def from_name(cls, plugin_name: str, **kwargs: Any) -> "YAMLEmbeddingClient":
        """Create an embedding client from plugin name."""
        spec = load_embedding_plugin(plugin_name)
        return cls(spec, **kwargs)

    @property
    def plugin_name(self) -> str:
        return self.spec.plugin_name

    def embed(self, text: str) -> list[float]:
        """Generate embedding for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        # Build payload based on input_wrap config
        input_config = self.spec.request

        if input_config.input_wrap == InputWrap.LIST:
            input_value = [text]
        elif input_config.input_wrap == InputWrap.STRING:
            input_value = text
        else:  # OBJECT
            input_value = {"text": text}

        payload = {input_config.input_field: input_value}

        # Add static fields
        payload.update(input_config.static_fields)

        # Add mapped parameters
        for fitz_name, provider_name in input_config.param_map.items():
            if fitz_name in self.params:
                payload[provider_name] = self.params[fitz_name]

        # Make request
        response = self._make_request(payload)

        # Extract embedding
        try:
            embedding = extract_path(response, self.spec.response.embeddings_path)
            return list(embedding)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to extract embedding: {e}. Response: {response}") from e


# =============================================================================
# YAML Rerank Plugin
# =============================================================================


class YAMLRerankClient(YAMLPluginBase):
    """
    Generic rerank plugin driven by YAML specification.

    Usage:
        client = YAMLRerankClient.from_name("cohere")
        results = client.rerank("query", ["doc1", "doc2", "doc3"])
    """

    plugin_type = "rerank"

    def __init__(self, spec: RerankPluginSpec, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)

    @classmethod
    def from_name(cls, plugin_name: str, **kwargs: Any) -> "YAMLRerankClient":
        """Create a rerank client from plugin name."""
        spec = load_rerank_plugin(plugin_name)
        return cls(spec, **kwargs)

    @property
    def plugin_name(self) -> str:
        return self.spec.plugin_name

    def rerank(
            self,
            query: str,
            documents: list[str],
            top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            top_n: Optional limit on results

        Returns:
            List of (index, score) tuples in ranked order
        """
        if not documents:
            return []

        req_config = self.spec.request

        payload = {
            req_config.query_field: query,
            req_config.documents_field: documents,
        }

        # Add static fields
        payload.update(req_config.static_fields)

        # Add mapped parameters
        params_with_top_n = {**self.params}
        if top_n is not None:
            params_with_top_n["top_n"] = top_n

        for fitz_name, provider_name in req_config.param_map.items():
            if fitz_name in params_with_top_n:
                payload[provider_name] = params_with_top_n[fitz_name]

        # Make request
        response = self._make_request(payload)

        # Extract results
        try:
            results = extract_path(response, self.spec.response.results_path)

            ranked: list[tuple[int, float]] = []
            for result in results:
                idx = extract_path(result, self.spec.response.result_index_path)
                score = extract_path(result, self.spec.response.result_score_path)
                ranked.append((int(idx), float(score)))

            return ranked

        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to extract rerank results: {e}. Response: {response}") from e


# =============================================================================
# Factory Functions
# =============================================================================


def create_yaml_chat_client(plugin_name: str, **kwargs: Any) -> YAMLChatClient:
    """Factory function to create a YAML chat client."""
    return YAMLChatClient.from_name(plugin_name, **kwargs)


def create_yaml_embedding_client(plugin_name: str, **kwargs: Any) -> YAMLEmbeddingClient:
    """Factory function to create a YAML embedding client."""
    return YAMLEmbeddingClient.from_name(plugin_name, **kwargs)


def create_yaml_rerank_client(plugin_name: str, **kwargs: Any) -> YAMLRerankClient:
    """Factory function to create a YAML rerank client."""
    return YAMLRerankClient.from_name(plugin_name, **kwargs)
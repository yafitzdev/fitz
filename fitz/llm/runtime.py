# fitz/llm/runtime.py
"""
Generic plugin runtime that executes YAML-defined plugins.

This is the "engine" that reads YAML plugin specs and performs the actual
HTTP calls, credential resolution, and response parsing.
"""
from __future__ import annotations

import logging
import os
from typing import Any, ClassVar, Literal, overload

from fitz.core.http import (
    APIError,
    HTTPClientNotAvailable,
    create_api_client,
    handle_api_error,
    raise_for_status,
)
from fitz.core.utils import extract_path, set_nested_path
from fitz.llm.credentials import CredentialError, resolve_api_key
from fitz.llm.loader import load_plugin
from fitz.llm.schema import (
    AuthType,
    ChatPluginSpec,
    EmbeddingPluginSpec,
    InputWrap,
    PluginSpec,
    RerankPluginSpec,
)
from fitz.llm.transforms import get_transformer

logger = logging.getLogger(__name__)


class YAMLPluginBase:
    """
    Base class for YAML-driven plugin clients.

    Handles: spec storage, parameter merging, env var resolution,
    API key resolution, HTTP client creation, request execution.
    """

    plugin_type: ClassVar[str] = ""

    def __init__(self, spec: PluginSpec, **kwargs: Any) -> None:
        self.spec = spec
        self.params = {**spec.defaults, **kwargs}

        # Resolve required env vars
        self._env_values: dict[str, str] = {}
        for req_env in spec.required_env:
            value = kwargs.get(req_env.inject_as) or os.getenv(req_env.name) or req_env.default
            if not value:
                raise ValueError(
                    f"{req_env.name} is required. "
                    f"Set it as an environment variable or pass {req_env.inject_as!r} parameter."
                )
            self._env_values[req_env.inject_as] = value

        self._base_url = self._resolve_placeholders(spec.provider.base_url)

        # Resolve API key
        self._api_key: str | None = None
        if spec.auth.type != AuthType.NONE:
            try:
                self._api_key = resolve_api_key(
                    provider=spec.provider.name,
                    config={"api_key": kwargs.get("api_key")} if kwargs.get("api_key") else None,
                )
            except CredentialError as e:
                raise RuntimeError(str(e)) from e

        self._client = self._create_client()

    @property
    def plugin_name(self) -> str:
        return self.spec.plugin_name

    def _resolve_placeholders(self, template: str) -> str:
        result = template
        for key, value in self._env_values.items():
            result = result.replace(f"{{{key}}}", value)
        for key, value in self.params.items():
            if isinstance(value, str):
                result = result.replace(f"{{{key}}}", value)
        return result

    def _create_client(self) -> Any:
        extra_headers: dict[str, str] = {}
        if self._api_key and self.spec.auth.type != AuthType.NONE:
            header_value = self.spec.auth.header_format.format(key=self._api_key)
            extra_headers[self.spec.auth.header_name] = header_value

        try:
            return create_api_client(
                base_url=self._base_url,
                api_key=None,
                timeout_type=self.spec.plugin_type,
                headers=extra_headers if extra_headers else None,
            )
        except HTTPClientNotAvailable:
            raise RuntimeError(
                "httpx is required for YAML plugins. Install with: pip install httpx"
            )

    def _make_request(self, payload: dict[str, Any]) -> dict[str, Any]:
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
        if hasattr(self, "_client"):
            try:
                self._client.close()
            except Exception:
                pass


class YAMLChatClient(YAMLPluginBase):
    """Chat plugin client."""

    plugin_type: ClassVar[str] = "chat"
    spec: ChatPluginSpec

    def __init__(self, spec: ChatPluginSpec, **kwargs: Any) -> None:
        super().__init__(spec, **kwargs)
        self._transformer = get_transformer(spec.request.messages_transform.value)

    def chat(self, messages: list[dict[str, Any]]) -> str:
        payload = self._transformer.transform(messages)
        payload.update(self.spec.request.static_fields)

        for fitz_name, provider_name in self.spec.request.param_map.items():
            if fitz_name in self.params:
                set_nested_path(payload, provider_name, self.params[fitz_name])

        response = self._make_request(payload)

        try:
            content = extract_path(response, self.spec.response.content_path)
            return str(content) if content else ""
        except (KeyError, IndexError) as e:
            logger.warning(f"Failed to extract response: {e}. Full response: {response}")
            return ""


class YAMLEmbeddingClient(YAMLPluginBase):
    """Embedding plugin client."""

    plugin_type: ClassVar[str] = "embedding"
    spec: EmbeddingPluginSpec

    def embed(self, text: str) -> list[float]:
        input_config = self.spec.request

        if input_config.input_wrap == InputWrap.LIST:
            input_value = [text]
        elif input_config.input_wrap == InputWrap.STRING:
            input_value = text
        else:
            input_value = {"text": text}

        payload = {input_config.input_field: input_value}
        payload.update(input_config.static_fields)

        for fitz_name, provider_name in input_config.param_map.items():
            if fitz_name in self.params:
                payload[provider_name] = self.params[fitz_name]

        response = self._make_request(payload)

        try:
            embedding = extract_path(response, self.spec.response.embeddings_path)
            return list(embedding)
        except (KeyError, IndexError) as e:
            raise RuntimeError(f"Failed to extract embedding: {e}. Response: {response}") from e


class YAMLRerankClient(YAMLPluginBase):
    """Rerank plugin client."""

    plugin_type: ClassVar[str] = "rerank"
    spec: RerankPluginSpec

    def rerank(
        self,
        query: str,
        documents: list[str],
        top_n: int | None = None,
    ) -> list[tuple[int, float]]:
        if not documents:
            return []

        req_config = self.spec.request
        payload = {
            req_config.query_field: query,
            req_config.documents_field: documents,
        }
        payload.update(req_config.static_fields)

        params_with_top_n = {**self.params}
        if top_n is not None:
            params_with_top_n["top_n"] = top_n

        for fitz_name, provider_name in req_config.param_map.items():
            if fitz_name in params_with_top_n:
                payload[provider_name] = params_with_top_n[fitz_name]

        response = self._make_request(payload)

        try:
            results = extract_path(response, self.spec.response.results_path)
            ranked: list[tuple[int, float]] = []
            for result in results:
                idx = extract_path(result, self.spec.response.result_index_path)
                score = extract_path(result, self.spec.response.result_score_path)
                ranked.append((int(idx), float(score)))
            return ranked
        except (KeyError, IndexError) as e:
            raise RuntimeError(
                f"Failed to extract rerank results: {e}. Response: {response}"
            ) from e


_CLIENT_CLASSES: dict[str, type[YAMLPluginBase]] = {
    "chat": YAMLChatClient,
    "embedding": YAMLEmbeddingClient,
    "rerank": YAMLRerankClient,
}


@overload
def create_yaml_client(
    plugin_type: Literal["chat"], plugin_name: str, **kwargs: Any
) -> YAMLChatClient: ...


@overload
def create_yaml_client(
    plugin_type: Literal["embedding"], plugin_name: str, **kwargs: Any
) -> YAMLEmbeddingClient: ...


@overload
def create_yaml_client(
    plugin_type: Literal["rerank"], plugin_name: str, **kwargs: Any
) -> YAMLRerankClient: ...


@overload
def create_yaml_client(plugin_type: str, plugin_name: str, **kwargs: Any) -> YAMLPluginBase: ...


def create_yaml_client(plugin_type: str, plugin_name: str, **kwargs: Any) -> YAMLPluginBase:
    """
    Create a YAML plugin client.

    Args:
        plugin_type: "chat", "embedding", or "rerank"
        plugin_name: Plugin name (e.g., "cohere", "openai")
        **kwargs: Plugin configuration

    Returns:
        Configured client instance
    """
    if plugin_type not in _CLIENT_CLASSES:
        raise ValueError(
            f"Invalid plugin type: {plugin_type!r}. "
            f"Must be one of: {sorted(_CLIENT_CLASSES.keys())}"
        )

    spec = load_plugin(plugin_type, plugin_name)
    client_class = _CLIENT_CLASSES[plugin_type]
    return client_class(spec, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "YAMLPluginBase",
    "YAMLChatClient",
    "YAMLEmbeddingClient",
    "YAMLRerankClient",
    "create_yaml_client",
]

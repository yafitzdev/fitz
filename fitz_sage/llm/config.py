# fitz_sage/llm/config.py
"""
Configuration parser for LLM providers.

There are two chat paths in fitz-sage:

``OnnxChat``
              the managed in-process Qwen3 0.6B ONNX GenAI enrichment runtime.
              This is the default for ingestion enrichment and needs no
              external inference server.
``OpenAICompatChat`` / ``OpenAICompatVision``
              the optional OpenAI HTTP protocol path for user-supplied
              synthesis, query intelligence, and vision endpoints
              (OpenAI, Azure, vLLM, LM Studio, Together, Fireworks, Groq,
              OpenRouter, Ollama in /v1 mode, …).

Provider names are configuration knobs over those implementations:

    endpoint  — bring your own URL + model. Default (and only) auth is
                NoAuth; opt-in to ApiKeyAuth via ``auth.api_key_env``.
    onnx      — managed local Qwen3 0.6B ONNX GenAI generation on CPU.
    openai    — preset for ``https://api.openai.com/v1`` + OPENAI_API_KEY,
                with default models from OPENAI_CHAT_MODELS.
    azure_openai
              — preset for Azure: requires ``base_url`` (Azure URL is
                tenant-specific) + AZURE_OPENAI_API_KEY.
    enterprise
              — separate path for OAuth2 + API-key composite auth
                (kept for the day-job/automotive deployment flow).

"""

from __future__ import annotations

import os
from typing import Any, Literal

from fitz_sage.llm.auth import ApiKeyAuth, AuthProvider, CompositeAuth, M2MAuth, NoAuth
from fitz_sage.llm.providers.base import (
    ChatProvider,
    ModelTier,
    RerankProvider,
    VisionProvider,
)

# Default OpenAI public API endpoint, used by the ``openai`` preset.
_OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"

# Provider name → environment variable mapping (None means no auth).
ENV_VAR_MAP: dict[str, str | None] = {
    "openai": "OPENAI_API_KEY",
    "azure_openai": "AZURE_OPENAI_API_KEY",
    "enterprise": None,  # auth configured explicitly via auth block
    "endpoint": None,  # NoAuth by default; opt-in via auth.api_key_env
}

# Provider name → header format for ApiKeyAuth.
HEADER_FORMAT_MAP: dict[str, str] = {
    "openai": "bearer",
    "azure_openai": "bearer",
}


def parse_provider_string(spec: str) -> tuple[str, str | None]:
    """
    Parse a provider/model spec string.

    Args:
        spec: Either ``provider`` or ``provider/model``. Models may
            contain forward slashes (only the first ``/`` splits).

    Returns:
        ``(provider_name, model_name_or_None)``
    """
    if "/" in spec:
        provider, model = spec.split("/", 1)
        return provider.strip(), model.strip()
    return spec.strip(), None


def _validate_enterprise_config(auth_config: dict[str, Any]) -> None:
    """Validate enterprise auth config with actionable error messages."""
    required = ["token_url", "client_id", "client_secret", "llm_api_key_env"]
    missing = [f for f in required if not auth_config.get(f)]

    if missing:
        raise ValueError(
            f"Enterprise auth config missing required fields: {', '.join(missing)}\n\n"
            f"Required configuration:\n"
            f"  auth:\n"
            f"    type: enterprise\n"
            f"    token_url: <OAuth2 token endpoint>\n"
            f"    client_id: ${{CLIENT_ID}}  # or literal value\n"
            f"    client_secret: ${{CLIENT_SECRET}}\n"
            f"    llm_api_key_env: <env var name for LLM API key>\n\n"
            f"Optional fields:\n"
            f"    scope: <OAuth2 scope>\n"
            f"    llm_api_key_header: X-Api-Key  # default\n"
            f"    client_cert_path: <mTLS cert path>\n"
            f"    client_key_path: <mTLS key path>\n"
            f"    client_key_password: ${{KEY_PASSWORD}}"
        )

    api_key_env = auth_config["llm_api_key_env"]
    if not os.environ.get(api_key_env):
        raise ValueError(
            f"Enterprise auth requires environment variable {api_key_env} to be set.\n"
            f"This variable should contain the LLM API key for the underlying provider."
        )


def resolve_auth(provider: str, config: dict[str, Any] | None = None) -> AuthProvider | None:
    """
    Resolve authentication for a provider.

    Args:
        provider: Provider name. Must be one of ``ENV_VAR_MAP``.
        config: Optional config dict. May contain an ``auth`` block.

    Returns:
        An ``AuthProvider`` instance, or ``None`` if the provider doesn't
        need auth (no current providers return None).

    Auth block formats:

    - **Endpoint with API key** (e.g. Together, Groq, Fireworks)::

          auth:
            api_key_env: TOGETHER_API_KEY
            header_format: bearer    # optional; default

    - **M2M OAuth2** (token auto-refreshes)::

          auth:
            type: m2m
            token_url: https://auth.corp.com/oauth/token
            client_id: ${CLIENT_ID}
            client_secret: ${CLIENT_SECRET}
            scope: optional-scope

    - **Enterprise** (M2M + downstream API key)::

          auth:
            type: enterprise
            token_url: ...
            client_id: ${CLIENT_ID}
            client_secret: ${CLIENT_SECRET}
            llm_api_key_env: BMW_LLM_API_KEY
            llm_api_key_header: X-Api-Key   # optional; default
            client_cert_path: ...           # optional, mTLS
            client_key_path: ...            # optional, mTLS
            client_key_password: ${KEY_PASSWORD}  # optional, mTLS
    """
    config = config or {}
    auth_config = config.get("auth", {})
    cert_path = config.get("cert_path") or auth_config.get("cert_path")

    # Enterprise composite auth (M2M + API key)
    if auth_config.get("type") == "enterprise":
        _validate_enterprise_config(auth_config)

        m2m = M2MAuth(
            token_url=auth_config["token_url"],
            client_id=auth_config["client_id"],
            client_secret=auth_config["client_secret"],
            cert_path=cert_path,
            scope=auth_config.get("scope"),
            client_cert_path=auth_config.get("client_cert_path"),
            client_key_path=auth_config.get("client_key_path"),
            client_key_password=auth_config.get("client_key_password"),
        )

        api_key_env = auth_config["llm_api_key_env"]
        header_name = auth_config.get("llm_api_key_header", "X-Api-Key")
        enterprise_header_format: Literal["bearer", "x-api-key", "basic"] = (
            "x-api-key" if header_name.lower() == "x-api-key" else "bearer"
        )
        api_key = ApiKeyAuth(api_key_env, header_format=enterprise_header_format)

        return CompositeAuth(m2m, api_key)

    # M2M auth on its own
    if auth_config.get("type") == "m2m":
        return M2MAuth(
            token_url=auth_config["token_url"],
            client_id=auth_config["client_id"],
            client_secret=auth_config["client_secret"],
            cert_path=cert_path,
            scope=auth_config.get("scope"),
        )

    # Enterprise provider name without an explicit auth block
    if provider == "enterprise":
        raise ValueError(
            "The 'enterprise' provider requires an 'auth' block in config.\n"
            "Example:\n"
            "  synthesizer: enterprise/gpt-4o\n"
            "  base_url: https://corp.gateway/openai/v1\n"
            "  auth:\n"
            "    type: enterprise\n"
            "    token_url: ...\n"
            "    client_id: ...\n"
            "    client_secret: ...\n"
            "    llm_api_key_env: ..."
        )

    # Endpoint: NoAuth by default, opt-in to ApiKeyAuth.
    if provider == "endpoint":
        if auth_config.get("api_key_env"):
            endpoint_header_format: Literal["bearer", "x-api-key", "basic"] = auth_config.get(
                "header_format", "bearer"
            )
            return ApiKeyAuth(auth_config["api_key_env"], header_format=endpoint_header_format)
        return NoAuth()

    # Named cloud presets: ApiKeyAuth from a known env var.
    env_var = ENV_VAR_MAP.get(provider)
    if env_var is not None:
        header_format: Literal["bearer", "x-api-key", "basic"] = HEADER_FORMAT_MAP.get(
            provider, "bearer"
        )  # type: ignore[assignment]
        return ApiKeyAuth(env_var, header_format=header_format)

    raise ValueError(
        f"Unknown provider: {provider}. "
        f"Supported: 'endpoint', 'openai', 'azure_openai', 'enterprise'."
    )


def _get_provider_kwargs(config: dict[str, Any] | None) -> dict[str, Any]:
    """Extract provider kwargs from config."""
    if not config:
        return {}

    kwargs: dict[str, Any] = {}

    if "base_url" in config:
        kwargs["base_url"] = config["base_url"]

    # Per-tier model override table
    if "models" in config:
        kwargs["models"] = config["models"]

    if "model" in config:
        kwargs["model"] = config["model"]

    if "num_ctx" in config:
        kwargs["num_ctx"] = config["num_ctx"]

    return kwargs


def _resolve_endpoint_kwargs(
    spec: str,
    config: dict[str, Any] | None,
    *,
    require_model: bool,
    role: str,
) -> tuple[AuthProvider, dict[str, Any]]:
    """
    Common kwarg/auth resolution for the ``endpoint`` preset.

    Returns ``(auth, kwargs)`` where ``kwargs`` always contains
    ``base_url`` and (if ``require_model``) ``model``.

    Raises ``ValueError`` with actionable messages if required fields
    are missing.
    """
    _, model = parse_provider_string(spec)
    auth = resolve_auth("endpoint", config)
    assert auth is not None  # endpoint always returns NoAuth or ApiKeyAuth
    kwargs = _get_provider_kwargs(config)
    if model:
        kwargs["model"] = model

    base_url = kwargs.get("base_url")
    if not base_url:
        raise ValueError(
            f"endpoint provider requires 'base_url' in config.\n"
            f"Example:\n"
            f"  {role}: endpoint/<model>\n"
            f"  base_url: http://localhost:8080/v1"
        )
    if require_model and not kwargs.get("model"):
        raise ValueError(
            f"endpoint provider requires a model in the spec.\n"
            f"Example:\n"
            f"  {role}: endpoint/<model>"
        )
    return auth, kwargs


def _resolve_openai_preset_kwargs(
    spec: str,
    config: dict[str, Any] | None,
    *,
    azure: bool,
) -> tuple[AuthProvider, dict[str, Any]]:
    """
    Resolve auth + kwargs for the ``openai`` / ``azure_openai`` preset.

    For ``openai``, base_url defaults to OpenAI's public API. For
    ``azure_openai``, the user must supply a base_url (it is
    tenant-specific).
    """
    provider, model = parse_provider_string(spec)
    auth = resolve_auth(provider, config)
    assert auth is not None  # presets always require an API key
    kwargs = _get_provider_kwargs(config)

    if model:
        kwargs["model"] = model

    if not kwargs.get("base_url"):
        if azure:
            raise ValueError(
                "azure_openai requires 'base_url' (Azure endpoints are "
                "tenant-specific).\nExample:\n"
                "  synthesizer: azure_openai/gpt-4o\n"
                "  base_url: https://my-tenant.openai.azure.com/openai/deployments/my-deployment"
            )
        kwargs["base_url"] = _OPENAI_DEFAULT_BASE_URL

    return auth, kwargs


def create_chat_provider(
    spec: str,
    config: dict[str, Any] | None = None,
    tier: ModelTier = "smart",
) -> ChatProvider:
    """
    Create a chat provider from a spec string.

    Args:
        spec: ``provider`` or ``provider/model`` (e.g. ``onnx/qwen3-0.6b``,
            ``endpoint/qwen2.5-7b``, ``openai/gpt-4o``,
            ``azure_openai/my-deployment``).
        config: Optional config dict (auth, base_url, etc.).
        tier: Tier hint when no model is supplied.

    Returns:
        A ChatProvider instance.
    """
    provider, _ = parse_provider_string(spec)
    if provider == "onnx":
        from fitz_sage.llm.providers.onnx_chat import DEFAULT_QWEN_MODEL_ALIAS, OnnxChat

        _, model = parse_provider_string(spec)
        return OnnxChat(model_id=model or DEFAULT_QWEN_MODEL_ALIAS)

    if provider == "enterprise":
        from fitz_sage.llm.providers.enterprise import EnterpriseChat

        auth = resolve_auth(provider, config)
        kwargs = _get_provider_kwargs(config)
        _, model = parse_provider_string(spec)
        if model:
            kwargs["model"] = model

        base_url = kwargs.pop("base_url", None)
        model_name = kwargs.pop("model", None)
        if not base_url:
            raise ValueError(
                "enterprise provider requires 'base_url' in config.\n"
                "Example:\n"
                "  synthesizer: enterprise/gpt-4o\n"
                "  base_url: https://corp.gateway/openai/v1"
            )
        if not model_name:
            raise ValueError(
                "enterprise provider requires a model in the spec.\n"
                "Example:\n"
                "  synthesizer: enterprise/gpt-4o"
            )
        return EnterpriseChat(auth, base_url=base_url, model=model_name, **kwargs)  # type: ignore[arg-type]

    if provider == "endpoint":
        from fitz_sage.llm.providers.openai_compat import OpenAICompatChat

        auth, kwargs = _resolve_endpoint_kwargs(
            spec, config, require_model=True, role="synthesizer"
        )
        base_url = kwargs.pop("base_url")
        model_name = kwargs.pop("model")
        return OpenAICompatChat(auth, model=model_name, base_url=base_url, tier=tier, **kwargs)

    if provider in ("openai", "azure_openai"):
        from fitz_sage.llm.providers.openai_compat import OpenAICompatChat

        auth, kwargs = _resolve_openai_preset_kwargs(
            spec, config, azure=(provider == "azure_openai")
        )
        return OpenAICompatChat(auth, tier=tier, **kwargs)

    raise ValueError(
        f"Unknown chat provider: {provider}. "
        f"Supported: 'onnx', 'endpoint', 'openai', 'azure_openai', 'enterprise'."
    )


def create_rerank_provider(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> RerankProvider | None:
    """Create a rerank provider from a spec string.

    The canonical backend is ``onnx`` — an INT8 ONNX cross-encoder
    loaded directly through ONNX Runtime. Default model is
    `Alibaba-NLP/gte-reranker-modernbert-base`. Override with
    ``onnx/<hf-model-id>`` to use a different cross-encoder.

    ``None`` is accepted only for direct low-level tests; engine config keeps
    reranking mandatory.
    """
    if spec is None:
        return None

    provider, model = parse_provider_string(spec)
    if provider == "onnx":
        from fitz_sage.llm.providers.onnx_reranker import DEFAULT_MODEL_ID, OnnxReranker

        return OnnxReranker(model_id=model or DEFAULT_MODEL_ID)

    raise ValueError(
        f"Unknown rerank provider: {provider}. " f"Supported: 'onnx' or 'onnx/<hf-model-id>'."
    )


def create_vision_provider(
    spec: str | None,
    config: dict[str, Any] | None = None,
) -> VisionProvider | None:
    """
    Create a vision provider from a spec string.

    Vision uses the same OpenAI-compatible chat-completions endpoint
    with image content parts; any vision-capable model behind an
    ``endpoint`` URL works.
    """
    if spec is None:
        return None

    provider, _ = parse_provider_string(spec)
    if provider == "endpoint":
        from fitz_sage.llm.providers.openai_compat import OpenAICompatVision

        auth, kwargs = _resolve_endpoint_kwargs(spec, config, require_model=True, role="vision")
        base_url = kwargs.pop("base_url")
        model_name = kwargs.pop("model")
        return OpenAICompatVision(auth, model=model_name, base_url=base_url, **kwargs)

    if provider in ("openai", "azure_openai"):
        from fitz_sage.llm.providers.openai_compat import OpenAICompatVision

        auth, kwargs = _resolve_openai_preset_kwargs(
            spec, config, azure=(provider == "azure_openai")
        )
        kwargs.pop("models", None)
        return OpenAICompatVision(auth, **kwargs)

    raise ValueError(
        f"Unknown vision provider: {provider}. " f"Supported: 'endpoint', 'openai', 'azure_openai'."
    )


__all__ = [
    "parse_provider_string",
    "resolve_auth",
    "create_chat_provider",
    "create_rerank_provider",
    "create_vision_provider",
    "ENV_VAR_MAP",
    "HEADER_FORMAT_MAP",
]

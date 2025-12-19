# fitz/llm/schema.py
"""
Pydantic schemas for YAML plugin validation.

These schemas ensure YAML plugin definitions are correct BEFORE runtime,
catching typos, missing fields, and invalid values early.

Schema hierarchy:
- BasePluginSpec: Common fields shared by all plugin types
  - ChatPluginSpec: Chat/completion plugins
  - EmbeddingPluginSpec: Text embedding plugins
  - RerankPluginSpec: Document reranking plugins
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# =============================================================================
# Enums
# =============================================================================


class AuthType(str, Enum):
    """Supported authentication types."""

    BEARER = "bearer"
    HEADER = "header"
    QUERY = "query"
    NONE = "none"


class InputWrap(str, Enum):
    """How to wrap input for embedding APIs."""

    LIST = "list"
    STRING = "string"
    OBJECT = "object"


class MessageTransform(str, Enum):
    """Predefined message transformation strategies."""

    OPENAI_CHAT = "openai_chat"
    COHERE_CHAT = "cohere_chat"
    ANTHROPIC_CHAT = "anthropic_chat"
    GEMINI_CHAT = "gemini_chat"
    OLLAMA_CHAT = "ollama_chat"


# =============================================================================
# Shared Configuration Components
# =============================================================================


class AuthConfig(BaseModel):
    """Authentication configuration for a provider."""

    model_config = ConfigDict(extra="forbid")

    type: AuthType = AuthType.BEARER
    header_name: str = "Authorization"
    header_format: str = "Bearer {key}"
    env_vars: list[str] = Field(default_factory=list)

    @field_validator("header_format")
    @classmethod
    def validate_header_format(cls, v: str) -> str:
        if "{key}" not in v and v != "":
            raise ValueError("header_format must contain {key} placeholder")
        return v


class ProviderConfig(BaseModel):
    """Provider connection details."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Provider name for credential resolution")
    base_url: str = Field(..., description="API base URL")

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, v: str) -> str:
        if not v.startswith(("http://", "https://", "{")):
            raise ValueError("base_url must start with http://, https://, or be a {placeholder}")
        return v.rstrip("/")


class EndpointConfig(BaseModel):
    """API endpoint configuration."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(..., description="API endpoint path")
    method: Literal["GET", "POST", "PUT", "DELETE"] = "POST"
    timeout: int = Field(default=30, ge=1, le=600, description="Request timeout in seconds")

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if not v.startswith("/"):
            raise ValueError("path must start with /")
        return v


class HealthCheckConfig(BaseModel):
    """Optional health check for local services."""

    model_config = ConfigDict(extra="forbid")

    path: str = "/health"
    method: Literal["GET", "POST"] = "GET"
    timeout: int = Field(default=2, ge=1, le=10)


class RequiredEnvConfig(BaseModel):
    """Additional required environment variables (e.g., for Azure)."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(..., description="Environment variable name")
    inject_as: str = Field(..., description="Placeholder name to inject into base_url/config")
    default: str | None = Field(default=None, description="Optional default value")


# =============================================================================
# Base Plugin Specification
# =============================================================================


class BasePluginSpec(BaseModel):
    """
    Base specification shared by all plugin types.

    Contains common fields that every plugin needs:
    - Identity: plugin_name, version
    - Connection: provider, auth, endpoint
    - Configuration: defaults, required_env

    Subclasses add:
    - plugin_type: Literal type discriminator
    - request: Type-specific request configuration
    - response: Type-specific response extraction
    """

    model_config = ConfigDict(extra="forbid")

    # Identity
    plugin_name: str = Field(..., min_length=1, max_length=50)
    version: str = "1.0"

    # Connection
    provider: ProviderConfig
    auth: AuthConfig = Field(default_factory=AuthConfig)
    endpoint: EndpointConfig

    # Configuration
    defaults: dict[str, Any] = Field(default_factory=dict)
    required_env: list[RequiredEnvConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_auth_env_vars(self) -> "BasePluginSpec":
        """Ensure auth has env_vars if type is not none."""
        if self.auth.type != AuthType.NONE and not self.auth.env_vars:
            raise ValueError(f"auth.env_vars required when auth.type is {self.auth.type}")
        return self


# =============================================================================
# Chat Plugin Schema
# =============================================================================


class ChatRequestConfig(BaseModel):
    """Request transformation for chat plugins."""

    model_config = ConfigDict(extra="forbid")

    messages_transform: MessageTransform = MessageTransform.OPENAI_CHAT
    static_fields: dict[str, Any] = Field(default_factory=dict)
    param_map: dict[str, str] = Field(
        default_factory=lambda: {
            "model": "model",
            "temperature": "temperature",
            "max_tokens": "max_tokens",
        }
    )


class ChatResponseConfig(BaseModel):
    """Response extraction for chat plugins."""

    model_config = ConfigDict(extra="forbid")

    content_path: str = Field(..., description="Path to response text")
    metadata_paths: dict[str, str] = Field(default_factory=dict)
    is_array: bool = False
    array_index: int = 0


class ChatPluginSpec(BasePluginSpec):
    """Complete specification for a chat plugin."""

    plugin_type: Literal["chat"] = "chat"
    health_check: HealthCheckConfig | None = None

    request: ChatRequestConfig
    response: ChatResponseConfig


# =============================================================================
# Embedding Plugin Schema
# =============================================================================


class EmbeddingRequestConfig(BaseModel):
    """Request configuration for embedding plugins."""

    model_config = ConfigDict(extra="forbid")

    input_field: str = Field(default="input", description="Field name for input text")
    input_wrap: InputWrap = Field(default=InputWrap.LIST, description="How to wrap input")

    static_fields: dict[str, Any] = Field(default_factory=dict)
    param_map: dict[str, str] = Field(
        default_factory=lambda: {
            "model": "model",
        }
    )


class EmbeddingResponseConfig(BaseModel):
    """Response extraction for embedding plugins."""

    model_config = ConfigDict(extra="forbid")

    embeddings_path: str = Field(..., description="Path to embedding vector(s)")
    is_array: bool = True
    array_index: int = 0


class EmbeddingPluginSpec(BasePluginSpec):
    """Complete specification for an embedding plugin."""

    plugin_type: Literal["embedding"] = "embedding"
    health_check: HealthCheckConfig | None = None

    request: EmbeddingRequestConfig
    response: EmbeddingResponseConfig


# =============================================================================
# Rerank Plugin Schema
# =============================================================================


class RerankRequestConfig(BaseModel):
    """Request configuration for rerank plugins."""

    model_config = ConfigDict(extra="forbid")

    query_field: str = Field(default="query", description="Field name for query")
    documents_field: str = Field(default="documents", description="Field name for documents")

    static_fields: dict[str, Any] = Field(default_factory=dict)
    param_map: dict[str, str] = Field(
        default_factory=lambda: {
            "model": "model",
            "top_n": "top_n",
        }
    )


class RerankResponseConfig(BaseModel):
    """Response extraction for rerank plugins."""

    model_config = ConfigDict(extra="forbid")

    results_path: str = Field(..., description="Path to results array")
    result_index_path: str = Field(default="index", description="Path to index within result")
    result_score_path: str = Field(
        default="relevance_score", description="Path to score within result"
    )


class RerankPluginSpec(BasePluginSpec):
    """Complete specification for a rerank plugin."""

    plugin_type: Literal["rerank"] = "rerank"

    request: RerankRequestConfig
    response: RerankResponseConfig


# =============================================================================
# Union type for any plugin spec
# =============================================================================

PluginSpec = ChatPluginSpec | EmbeddingPluginSpec | RerankPluginSpec


__all__ = [
    # Enums
    "AuthType",
    "InputWrap",
    "MessageTransform",
    # Shared configs
    "AuthConfig",
    "ProviderConfig",
    "EndpointConfig",
    "HealthCheckConfig",
    "RequiredEnvConfig",
    # Base spec
    "BasePluginSpec",
    # Chat
    "ChatRequestConfig",
    "ChatResponseConfig",
    "ChatPluginSpec",
    # Embedding
    "EmbeddingRequestConfig",
    "EmbeddingResponseConfig",
    "EmbeddingPluginSpec",
    # Rerank
    "RerankRequestConfig",
    "RerankResponseConfig",
    "RerankPluginSpec",
    # Union
    "PluginSpec",
]

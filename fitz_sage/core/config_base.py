# fitz_sage/core/config_base.py
"""
Unified base configuration classes to reduce duplication.

Provides common patterns for:
- Timeout configuration
- Connection pooling
- API configuration
- Feature toggles
- Validation patterns
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class TimeoutMixin(BaseModel):
    """
    Common timeout configuration pattern.

    Used by: cloud, http clients, database connections, LLM providers.
    """

    timeout: int = Field(default=30, ge=1, le=600, description="Request timeout in seconds")

    connect_timeout: Optional[int] = Field(
        default=5, ge=1, le=30, description="Connection timeout in seconds"
    )


class PoolConfigMixin(BaseModel):
    """
    Database/connection pool configuration pattern.

    Used by: storage, vector_db, any pooled resource.
    """

    pool_min_size: int = Field(default=1, ge=1, le=100, description="Minimum connections in pool")

    pool_max_size: int = Field(default=10, ge=1, le=100, description="Maximum connections in pool")

    @field_validator("pool_max_size")
    def validate_pool_sizes(cls, v, info):
        """Ensure max >= min."""
        if info.data and "pool_min_size" in info.data:
            if v < info.data["pool_min_size"]:
                raise ValueError("pool_max_size must be >= pool_min_size")
        return v


class ApiConfigMixin(BaseModel):
    """
    External API configuration pattern.

    Used by: cloud, LLM providers, external services.
    """

    api_key: Optional[str] = Field(default=None, description="API key for authentication")

    base_url: Optional[str] = Field(default=None, description="API base URL")

    headers: dict[str, str] = Field(default_factory=dict, description="Additional HTTP headers")


class FeatureToggleMixin(BaseModel):
    """
    Feature enable/disable pattern.

    Used by: cloud, optional features, experimental flags.
    """

    enabled: bool = Field(default=False, description="Enable this feature")

    def is_enabled(self) -> bool:
        """Check if feature is enabled (for consistent naming)."""
        return self.enabled


class RetryConfigMixin(BaseModel):
    """
    Retry configuration pattern.

    Used by: HTTP clients, database operations, API calls.
    """

    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")

    retry_delay: float = Field(
        default=1.0, ge=0.1, le=60.0, description="Initial retry delay in seconds"
    )

    retry_backoff: float = Field(
        default=2.0, ge=1.0, le=5.0, description="Exponential backoff multiplier"
    )


class ValidationMixin(BaseModel):
    """
    Base class for configs that need validation.

    Provides consistent validation pattern.
    """

    def validate_config(self) -> None:
        """
        Validate configuration consistency.

        Override in subclasses to add specific validation logic.
        Raises ValueError if config is invalid.
        """
        pass

    def model_post_init(self, __context) -> None:
        """Automatically run validation after initialization."""
        self.validate_config()


# Composite base classes for common combinations


class ApiServiceConfig(ApiConfigMixin, TimeoutMixin, ValidationMixin):
    """
    Base for external API service configurations.

    Combines API settings, timeouts, and validation.
    Example: cloud config, LLM provider configs.
    """

    pass


class DatabaseConfig(PoolConfigMixin, TimeoutMixin, ValidationMixin):
    """
    Base for database/storage configurations.

    Combines connection pooling, timeouts, and validation.
    Example: storage config.
    """


class FeatureConfig(FeatureToggleMixin, ValidationMixin):
    """
    Base for feature flag configurations.

    Combines enable/disable with validation.
    Example: experimental features, optional modules.
    """

    pass


# Standard timeout values for consistency
class TimeoutDefaults(Enum):
    """Standard timeout values across the platform."""

    FAST = 5  # Health checks, simple queries
    NORMAL = 30  # Standard API calls
    SLOW = 60  # File operations, complex queries
    CHAT = 300  # LLM chat completions
    EMBED = 120  # Embedding operations

    @classmethod
    def get(cls, operation: str) -> int:
        """Get timeout for operation type."""
        mapping = {
            "health": cls.FAST.value,
            "query": cls.NORMAL.value,
            "chat": cls.CHAT.value,
            "embed": cls.EMBED.value,
            "file": cls.SLOW.value,
        }
        return mapping.get(operation, cls.NORMAL.value)

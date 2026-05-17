# fitz_sage/core/timeouts.py
"""
Centralized timeout configuration.

All timeout values should come from here instead of being hardcoded.
This ensures consistency and makes tuning easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True)
class Timeouts:
    """
    Central timeout configuration for the platform.

    All timeouts in seconds.
    """

    # Connection timeouts
    CONNECT: ClassVar[int] = 5
    CONNECT_DB: ClassVar[int] = 30

    # Health check timeouts
    HEALTH_CHECK: ClassVar[int] = 5
    OLLAMA_HEALTH: ClassVar[int] = 10

    # API operation timeouts
    API_DEFAULT: ClassVar[int] = 30
    API_FAST: ClassVar[int] = 10
    API_SLOW: ClassVar[int] = 60

    # LLM operation timeouts
    LLM_CHAT: ClassVar[int] = 300  # 5 minutes for chat
    LLM_RERANK: ClassVar[int] = 60  # 1 minute for reranking
    LLM_VISION: ClassVar[int] = 180  # 3 minutes for vision

    # Database operation timeouts
    DB_QUERY: ClassVar[int] = 30
    DB_BATCH: ClassVar[int] = 60
    DB_INIT: ClassVar[int] = 60

    # File operation timeouts
    FILE_READ: ClassVar[int] = 30
    FILE_PARSE: ClassVar[int] = 180  # 3 minutes for document parsing
    FILE_INGEST: ClassVar[int] = 300  # 5 minutes for ingestion

    # Background worker timeouts
    WORKER_JOIN: ClassVar[int] = 5
    WORKER_POLL: ClassVar[float] = 0.5

    # Retry timeouts
    RETRY_INITIAL: ClassVar[float] = 1.0
    RETRY_MAX: ClassVar[int] = 30

    @classmethod
    def for_operation(cls, operation: str) -> int:
        """
        Get timeout for a named operation.

        Args:
            operation: Operation name (e.g., "chat", "health")

        Returns:
            Timeout in seconds

        Examples:
            >>> Timeouts.for_operation("chat")
            300
            >>> Timeouts.for_operation("health")
            5
        """
        mapping = {
            # API operations
            "api": cls.API_DEFAULT,
            "api_fast": cls.API_FAST,
            "api_slow": cls.API_SLOW,
            # LLM operations
            "chat": cls.LLM_CHAT,
            "rerank": cls.LLM_RERANK,
            "vision": cls.LLM_VISION,
            # Database
            "db": cls.DB_QUERY,
            "db_batch": cls.DB_BATCH,
            "db_init": cls.DB_INIT,
            # Files
            "file": cls.FILE_READ,
            "parse": cls.FILE_PARSE,
            "ingest": cls.FILE_INGEST,
            # Health checks
            "health": cls.HEALTH_CHECK,
            "ollama_health": cls.OLLAMA_HEALTH,
            # Connection
            "connect": cls.CONNECT,
            "connect_db": cls.CONNECT_DB,
            # Workers
            "worker_join": cls.WORKER_JOIN,
        }
        return mapping.get(operation, cls.API_DEFAULT)


# Global instance for easy import
TIMEOUTS = Timeouts()


# Helper functions for common patterns
def get_httpx_timeout(operation: str = "api") -> "httpx.Timeout":  # noqa: F821
    """
    Get httpx.Timeout object for operation.

    Args:
        operation: Operation type

    Returns:
        httpx.Timeout configured for operation

    Example:
        >>> client = httpx.Client(timeout=get_httpx_timeout("chat"))
    """
    import httpx

    timeout = TIMEOUTS.for_operation(operation)
    connect = TIMEOUTS.CONNECT

    # Special handling for long operations
    if operation in ("chat", "vision", "ingest"):
        return httpx.Timeout(timeout, connect=connect)

    return httpx.Timeout(timeout, connect=connect)


def get_db_timeout(operation: str = "db") -> dict[str, float]:
    """
    Get database timeout configuration.

    Args:
        operation: Database operation type

    Returns:
        Dict with timeout and connect_timeout

    Example:
        >>> pool = AsyncConnectionPool(**get_db_timeout("db_batch"))
    """
    return {
        "timeout": TIMEOUTS.for_operation(operation),
        "connect_timeout": TIMEOUTS.CONNECT_DB,
    }

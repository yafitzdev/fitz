# benchmarks/corpora/core/code/auth_service.py
"""Authentication service used by benchmark cases."""


class AuthService:
    """Session authentication service."""

    def refresh_expired_session(self, session_id: str) -> bool:
        """Refresh an expired session when the session is inside the grace window."""
        return session_id.startswith("grace-")

    def revoke_session(self, session_id: str) -> str:
        """Revoke any active or expired session."""
        return f"revoked:{session_id}"


class SessionExpiredError(RuntimeError):
    """Raised when a session expired outside the refresh grace window."""

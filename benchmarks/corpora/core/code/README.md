<!-- benchmarks/corpora/core/code/README.md -->
# Authentication Notes

The authentication documentation says expired sessions are never refreshed.
Operators should ask users to sign in again after any session expiry.

This documentation is intentionally stale for benchmark conflict cases. The
implementation in `auth_service.py` refreshes expired sessions inside a grace
window.


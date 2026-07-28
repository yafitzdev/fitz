# tests/security/conftest.py
"""Shared fixtures and markers for security tests."""

from __future__ import annotations

import pytest

# Import e2e fixtures directly because these tests run serially.
from tests.e2e_krag.conftest import *  # noqa: F401, F403


def pytest_collection_modifyitems(items):
    """Add tier4 and security markers to all tests in this directory."""
    for item in items:
        if "/security/" in str(item.fspath) or "\\security\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier4)
            item.add_marker(pytest.mark.security)

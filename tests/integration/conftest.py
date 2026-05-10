# tests/integration/conftest.py
"""
Integration test fixtures.

These tests require real services (postgres, real LLM endpoints, etc.).
"""

from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )


def pytest_collection_modifyitems(items):
    """Add tier3 and integration markers to all tests in this directory."""
    for item in items:
        if "/integration/" in str(item.fspath) or "\\integration\\" in str(item.fspath):
            item.add_marker(pytest.mark.tier3)
            item.add_marker(pytest.mark.integration)

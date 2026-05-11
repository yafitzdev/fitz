# tests/conftest.py
"""
Root conftest - imports fixtures from all test modules.

All tests use the centralized tests/test_config.yaml.
Config structure matches .fitz/config/ format.

Test Tiers (for CI/CD optimization):
=====================================
- tier1: Critical path tests - pure logic, no I/O (<30s)
         Run: pytest -m tier1
- tier2: Unit tests with mocks - no real services (<2min)
         Run: pytest -m "tier1 or tier2"
- tier3: Integration tests - real LLM endpoints (<10min)
         Run: pytest -m "tier1 or tier2 or tier3"
- tier4: Heavy tests - security, chaos, load, performance (30min+)
         Run: pytest (all tests)

Recommended CI Configuration:
- Every commit:    pytest -m tier1
- PR merge:        pytest -m "tier1 or tier2"
- Merge to main:   pytest -m "tier1 or tier2 or tier3"
- Nightly:         pytest

Feature Markers:
- sqlite: SQLite-specific tests (connection manager, table store, FTS5)
- llm: Tests requiring real LLM API calls
- integration: Tests requiring real services
- e2e: End-to-end tests
- slow: Slow tests (>10s)
- security/chaos/performance/scalability: Category markers
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pytest
import yaml

# =============================================================================
# Test Configuration
# =============================================================================

TEST_CONFIG_PATH = Path(__file__).parent / "test_config.yaml"


@lru_cache(maxsize=1)
def load_test_config() -> dict:
    """Load the centralized test configuration."""
    with open(TEST_CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_test_chat(tier: str = "smart"):
    """
    Get chat client configured for tests (local Ollama).

    Args:
        tier: Model tier - "smart", "fast", or "balanced"
    """
    from fitz_sage.llm import get_chat

    config = load_test_config()
    # Get chat config from first tier (local)
    first_tier = config["tiers"][0]
    chat_spec = first_tier["chat"]
    chat_models = first_tier.get("chat_models", {})
    if chat_models.get(tier) and "/" not in chat_spec:
        chat_spec = f"{chat_spec}/{chat_models[tier]}"
    return get_chat(chat_spec, tier=tier)


@pytest.fixture
def test_chat():
    """Fixture providing the test chat client (local Ollama, smart tier)."""
    return get_test_chat("smart")


@pytest.fixture
def test_config():
    """Fixture providing the full test config dict."""
    return load_test_config()


# =============================================================================
# Import fixtures from submodules
# =============================================================================

# Note: E2E fixtures NOT imported here - they have autouse=True session fixtures
# that conflict with pytest-xdist parallel execution. E2E tests get their fixtures
# from tests/e2e/conftest.py directly via pytest's conftest discovery.

# Import unit test fixtures
from tests.unit.conftest import *  # noqa: E402, F401, F403

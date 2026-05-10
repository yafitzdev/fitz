# tests/test_default_config_loads_and_validates_minimally.py
"""
Test that the default config can be loaded and validated.
"""

from unittest.mock import patch

from fitz_sage.config import load_engine_config
from fitz_sage.engines.fitz_krag.config import FitzKragConfig


def test_default_config_loads_and_validates_base_schema():
    """
    Verifies the CURRENT architectural contract:

    - default.yaml can be loaded
    - resolved runtime config validates against FitzKragConfig
    - all required fields are present
    """
    # Mock _load_user_config to avoid user config interference
    with patch("fitz_sage.config.loader._load_user_config", return_value=None):
        cfg = load_engine_config("fitz_krag")

    assert isinstance(cfg, FitzKragConfig)

    # Core plugins (string specs)
    assert cfg.chat_fast
    assert cfg.chat_balanced
    assert cfg.chat_smart

    # Collection is required
    assert cfg.collection

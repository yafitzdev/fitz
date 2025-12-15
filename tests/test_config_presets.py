# tests/test_config_presets.py
from fitz.core.config.presets import get_preset
import pytest


def test_get_preset_local():
    config = get_preset("local")
    assert config["llm"]["plugin_name"] == "local"
    assert config["embedding"]["plugin_name"] == "local"
    assert config["rerank"]["plugin_name"] == "local"


def test_get_preset_dev():
    config = get_preset("dev")
    assert config["llm"]["plugin_name"] == "openai"
    assert config["llm"]["kwargs"]["model"] == "gpt-4o-mini"


def test_get_preset_production():
    config = get_preset("production")
    assert config["llm"]["plugin_name"] == "anthropic"
    assert config["embedding"]["plugin_name"] == "cohere"


def test_get_preset_invalid():
    with pytest.raises(ValueError, match="Unknown preset"):
        get_preset("invalid")


def test_get_preset_returns_copy():
    """Ensure preset dict is not mutated."""
    config1 = get_preset("local")
    config2 = get_preset("local")

    config1["llm"]["plugin_name"] = "modified"

    # config2 should be unchanged
    assert config2["llm"]["plugin_name"] == "local"
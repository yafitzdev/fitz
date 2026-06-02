# tests/unit/test_config_loader.py
"""Test config loading."""

from unittest.mock import patch

import pytest

from fitz_sage.config.loader import load_engine_config


def test_load_config_from_defaults():
    """Test loading config from default.yaml (isolated from user config)."""
    # Mock _load_user_config to return None, ensuring we only test defaults
    with patch("fitz_sage.config.loader._load_user_config", return_value=None):
        config = load_engine_config("fitz_krag")

    # Verify it's a Pydantic model with flat chat tier fields
    assert hasattr(config, "chat_fast")
    assert hasattr(config, "chat_balanced")
    assert hasattr(config, "chat_smart")
    assert hasattr(config, "collection")

    # Chat tier specs are optional; retrieval defaults do not require them.
    assert config.chat_smart is None

    # Verify None for disabled features
    assert config.vision is None or isinstance(config.vision, str)

    # Verify governance defaults to the pyrrho classifier
    assert config.governance == "pyrrho"

    # Verify generation settings
    assert hasattr(config, "enable_citations")
    assert hasattr(config, "strict_grounding")
    assert hasattr(config, "top_addresses")


def test_config_required_field():
    """Test that collection field is required."""
    from pydantic import ValidationError

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

    with pytest.raises(ValidationError):
        FitzKragConfig()


def test_config_validation():
    """Test config validation (Pydantic)."""
    from pydantic import ValidationError

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

    with pytest.raises(ValidationError):
        FitzKragConfig(collection="test", top_addresses=0)

    with pytest.raises(ValidationError):
        FitzKragConfig(collection="test", max_context_tokens=10)


def test_config_none_for_disabled():
    """Test that None properly disables optional features."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

    config = FitzKragConfig(
        collection="test",
        rerank=None,  # Explicitly disabled
        vision=None,  # Explicitly disabled
    )

    assert config.rerank is None
    assert config.vision is None


def test_enable_guardrails_raises_migration_error():
    """A config using the removed `enable_guardrails` key gets an actionable error."""
    with patch(
        "fitz_sage.config.loader._load_user_config",
        return_value={"enable_guardrails": False},
    ):
        with pytest.raises(ValueError, match="enable_guardrails"):
            load_engine_config("fitz_krag")


def test_create_governance_dispatch():
    """`create_governance` maps a config spec to a classifier instance or None."""
    from fitz_sage.governance import Pyrrho, create_governance
    from fitz_sage.governance.pyrrho import MODEL_ID, TAU

    assert create_governance(None) is None

    default = create_governance("pyrrho")
    assert isinstance(default, Pyrrho)
    assert default._model_id == MODEL_ID
    assert MODEL_ID == "yafitzdev/pyrrho-nano-g3"
    assert TAU == 0.60

    custom = create_governance("pyrrho/acme/custom-fine-tune")
    assert isinstance(custom, Pyrrho)
    assert custom._model_id == "acme/custom-fine-tune"


def test_create_governance_unknown_provider():
    """An unknown governance provider raises an actionable error."""
    from fitz_sage.governance import create_governance

    with pytest.raises(ValueError, match="Unknown governance provider"):
        create_governance("bogus")

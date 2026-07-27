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
    assert hasattr(config, "strict_grounding")
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


def test_config_rejects_nullable_mandatory_retrieval_features():
    """Governance and rerank are mandatory product features."""
    from pydantic import ValidationError

    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

    with pytest.raises(ValidationError):
        FitzKragConfig(collection="test", rerank=None)

    with pytest.raises(ValidationError):
        FitzKragConfig(collection="test", governance=None)


def test_config_none_for_optional_vision():
    """Vision remains optional because parser choice controls whether it is used."""
    from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

    config = FitzKragConfig(
        collection="test",
        vision=None,  # Explicitly disabled
    )

    assert config.vision is None


def test_create_pyrrho_dispatch(tmp_path):
    """`create_pyrrho` maps a config spec to the independent runtime."""
    from pyrrho import DEFAULT_MODEL_ID, DEFAULT_SUFFICIENT_THRESHOLD, Pyrrho

    from fitz_sage.integrations.pyrrho import create_pyrrho

    default = create_pyrrho("pyrrho")
    assert isinstance(default, Pyrrho)
    assert default.model_spec == DEFAULT_MODEL_ID
    assert DEFAULT_MODEL_ID == "yafitzdev/pyrrho-v2-nano-g1"
    assert DEFAULT_SUFFICIENT_THRESHOLD == 0.34

    custom = create_pyrrho("pyrrho/acme/custom-fine-tune")
    assert isinstance(custom, Pyrrho)
    assert custom.model_spec == "acme/custom-fine-tune"

    local_package = tmp_path / "pyrrho-v2-nano-g1"
    local_package.mkdir()
    local = create_pyrrho(f"pyrrho/{local_package}")
    assert isinstance(local, Pyrrho)
    assert local.model_spec == str(local_package)


def test_create_pyrrho_unknown_provider():
    """An unknown provider raises an actionable error."""
    from fitz_sage.integrations.pyrrho import create_pyrrho

    with pytest.raises(ValueError, match="Unknown governance provider"):
        create_pyrrho("bogus")

    with pytest.raises(ValueError, match="Governance must"):
        create_pyrrho(None)

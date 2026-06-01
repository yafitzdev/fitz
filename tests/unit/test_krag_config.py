# tests/unit/test_krag_config.py
"""Tests for FitzKragConfig schema and defaults."""

import pytest
import yaml

from fitz_sage.engines.fitz_krag.config import FitzKragConfig, get_default_config_path


class TestFitzKragConfig:
    def test_minimal_config(self):
        config = FitzKragConfig(collection="test")
        assert config.collection == "test"
        assert config.chat_fast is None
        assert config.chat_balanced is None
        assert config.chat_smart is None
        assert config.chat_base_url is None
        assert config.auth is None
        assert config.cert_path is None

    def test_defaults(self):
        config = FitzKragConfig(collection="test")
        assert config.code_languages == ["python", "typescript", "java", "go"]
        assert config.summary_batch_size == 15
        assert config.top_addresses == 50
        assert config.top_read == 50
        assert config.keyword_weight == 0.4
        assert config.enable_citations is True
        assert config.strict_grounding is True
        assert config.max_context_tokens == 48000
        assert config.max_answer_tokens == 512
        assert config.short_answer_tokens == 192

    def test_custom_values(self):
        """Cloud config: openai preset with API key in env."""
        config = FitzKragConfig(
            collection="my_project",
            chat_smart="openai/gpt-4o",
            chat_fast="openai/gpt-4o-mini",
            chat_balanced="openai/gpt-4o-mini",
            chat_base_url=None,
            top_addresses=20,
            keyword_weight=0.3,
        )
        assert config.chat_smart == "openai/gpt-4o"
        assert config.chat_fast == "openai/gpt-4o-mini"
        assert config.top_addresses == 20
        assert config.keyword_weight == 0.3

    def test_auth_config_allowed(self):
        """KRAG configs can pass auth blocks through to role providers."""
        config = FitzKragConfig(
            collection="enterprise_project",
            synthesizer="enterprise/openai/gpt-4o",
            chat_base_url="https://llm.corp.internal/v1",
            auth={
                "type": "enterprise",
                "token_url": "https://auth.corp.internal/oauth/token",
                "client_id": "${CLIENT_ID}",
                "client_secret": "${CLIENT_SECRET}",
                "llm_api_key_env": "CORP_LLM_API_KEY",
            },
            cert_path="/etc/ssl/corp-ca-bundle.crt",
        )

        assert config.auth is not None
        assert config.auth["type"] == "enterprise"
        assert config.cert_path == "/etc/ssl/corp-ca-bundle.crt"

    def test_collection_required(self):
        with pytest.raises(Exception):
            FitzKragConfig()  # type: ignore[call-arg]

    def test_validation_top_addresses(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", top_addresses=0)

    def test_validation_weights(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", keyword_weight=1.5)

    def test_extra_fields_forbidden(self):
        with pytest.raises(Exception):
            FitzKragConfig(collection="test", nonexistent_field=True)

    def test_no_chat_kwargs_field(self):
        """chat_kwargs, embedding_kwargs, rerank_kwargs, vision_kwargs are deleted."""
        config = FitzKragConfig(collection="test")
        assert not hasattr(config, "chat_kwargs")
        assert not hasattr(config, "embedding_kwargs")
        assert not hasattr(config, "rerank_kwargs")
        assert not hasattr(config, "vision_kwargs")


class TestDefaultYaml:
    def test_default_config_path_exists(self):
        path = get_default_config_path()
        assert path.exists()
        assert path.name == "default.yaml"

    def test_default_yaml_loads(self):
        path = get_default_config_path()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        assert "fitz_krag" in raw
        assert raw["fitz_krag"]["chat_fast"] is None
        assert raw["fitz_krag"]["chat_balanced"] is None
        assert raw["fitz_krag"]["chat_smart"] is None
        assert raw["fitz_krag"]["chat_base_url"] is None
        assert raw["fitz_krag"]["auth"] is None
        assert raw["fitz_krag"]["cert_path"] is None
        assert raw["fitz_krag"]["short_answer_tokens"] == 192
        assert raw["fitz_krag"]["collection"] == "default"
        # Embedding fields are gone — fitz-sage no longer uses dense vectors.
        assert "embedding" not in raw["fitz_krag"]
        assert "embedding_base_url" not in raw["fitz_krag"]

    def test_default_yaml_creates_valid_config(self):
        path = get_default_config_path()
        with path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        config = FitzKragConfig(**raw["fitz_krag"])
        assert config.collection == "default"

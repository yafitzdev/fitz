# tests/unit/test_cli_init.py
"""
Tests for config generation functions (init_config module).

The `fitz init` CLI command has been removed. These tests verify
the config generation functions still produce valid YAML.
"""

from __future__ import annotations


class TestGenerateConfig:
    """Tests for config generation functions."""

    def test_generate_global_config(self):
        """Test generate_global_config produces valid YAML."""
        import yaml

        from fitz_sage.cli.commands.init_config import generate_global_config

        config_str = generate_global_config("fitz_krag")
        config = yaml.safe_load(config_str)

        assert config["default_engine"] == "fitz_krag"

    def test_generate_fitz_krag_config_basic(self):
        """Test generate_fitz_krag_config produces valid YAML with flat format."""
        import yaml

        from fitz_sage.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            rerank=None,
            rerank_model="",
            vector_db="pgvector",
        )

        config = yaml.safe_load(config_str)

        # Flat format: tier-specific chat specs with provider/model
        assert "chat_fast" in config
        assert "chat_smart" in config
        assert config["vector_db"] == "pgvector"

    def test_generate_fitz_krag_config_with_rerank(self):
        """Test generate_fitz_krag_config includes rerank when provided."""
        import yaml

        from fitz_sage.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="command-a-03-2025",
            chat_model_fast="command-r7b-12-2024",
            rerank="cohere",
            rerank_model="rerank-v3.5",
            vector_db="pgvector",
        )

        config = yaml.safe_load(config_str)

        # Rerank includes model in provider/model format
        assert config["rerank"] == "cohere/rerank-v3.5"

    def test_generate_fitz_krag_config_without_rerank(self):
        """Test generate_fitz_krag_config sets rerank to null when not provided."""
        import yaml

        from fitz_sage.cli.commands.init_config import generate_fitz_krag_config

        config_str = generate_fitz_krag_config(
            chat="cohere",
            chat_model_smart="",
            chat_model_fast="",
            rerank=None,
            rerank_model="",
            vector_db="pgvector",
        )

        config = yaml.safe_load(config_str)

        assert config["rerank"] is None

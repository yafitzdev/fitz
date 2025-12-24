# tests/test_plugin_system.py
"""
Comprehensive tests for the plugin system.

Tests:
1. Schema defaults loading from YAML
2. Chat plugin loading and validation
3. Embedding plugin loading and validation
4. Rerank plugin loading and validation
5. Vector DB plugin loading (FAISS)

Run with: pytest tests/test_plugin_system.py -v
"""

import pytest

# =============================================================================
# Schema Defaults Tests
# =============================================================================


class TestSchemaDefaults:
    """Test schema_defaults.py functionality."""

    def test_load_chat_schema(self):
        """Test loading chat plugin schema."""
        from fitz_ai.llm.schema_defaults import get_all_fields

        fields = get_all_fields("chat")

        assert "plugin_name" in fields
        assert "plugin_type" in fields
        assert "provider.name" in fields
        assert "provider.base_url" in fields
        assert "auth.type" in fields
        assert "endpoint.path" in fields
        assert "request.messages_transform" in fields
        assert "response.content_path" in fields

    def test_load_embedding_schema(self):
        """Test loading embedding plugin schema."""
        from fitz_ai.llm.schema_defaults import get_all_fields

        fields = get_all_fields("embedding")

        assert "plugin_name" in fields
        assert "request.input_field" in fields
        assert "response.embeddings_path" in fields

    def test_load_rerank_schema(self):
        """Test loading rerank plugin schema."""
        from fitz_ai.llm.schema_defaults import get_all_fields

        fields = get_all_fields("rerank")

        assert "plugin_name" in fields
        assert "request.query_field" in fields
        assert "response.results_path" in fields

    def test_get_required_fields(self):
        """Test getting required fields."""
        from fitz_ai.llm.schema_defaults import get_required_fields

        required = get_required_fields("chat")

        assert "plugin_name" in required
        assert "plugin_type" in required
        assert "provider.name" in required
        assert "provider.base_url" in required
        assert "endpoint.path" in required
        assert "request.messages_transform" in required
        assert "response.content_path" in required

    def test_get_defaults(self):
        """Test getting default values."""
        from fitz_ai.llm.schema_defaults import get_defaults

        defaults = get_defaults("chat")

        assert defaults.get("version") == "1.0"
        assert defaults.get("auth.type") == "bearer"
        assert defaults.get("auth.header_name") == "Authorization"
        assert defaults.get("auth.header_format") == "Bearer {key}"
        assert defaults.get("endpoint.method") == "POST"
        assert defaults.get("endpoint.timeout") == 120

    def test_get_nested_defaults(self):
        """Test getting defaults as nested dict."""
        from fitz_ai.llm.schema_defaults import get_nested_defaults

        nested = get_nested_defaults("chat")

        assert nested.get("version") == "1.0"
        assert nested.get("auth", {}).get("type") == "bearer"
        assert nested.get("endpoint", {}).get("method") == "POST"

    def test_get_field_options(self):
        """Test getting field options."""
        from fitz_ai.llm.schema_defaults import get_field_options

        auth_options = get_field_options("chat", "auth.type")
        assert auth_options == ["bearer", "header", "query", "none"]

        transform_options = get_field_options("chat", "request.messages_transform")
        assert "openai_chat" in transform_options
        assert "cohere_chat" in transform_options

    def test_validate_plugin_fields(self):
        """Test plugin field validation."""
        from fitz_ai.llm.schema_defaults import validate_plugin_fields

        # Valid plugin data
        valid_data = {
            "plugin_name": "test",
            "plugin_type": "chat",
            "provider": {"name": "test", "base_url": "https://api.test.com"},
            "endpoint": {"path": "/chat"},
            "request": {"messages_transform": "openai_chat"},
            "response": {"content_path": "text"},
        }

        errors = validate_plugin_fields("chat", valid_data)
        # Should have no errors for required fields
        required_errors = [e for e in errors if "Missing required" in e]
        assert len(required_errors) == 0

    def test_invalid_plugin_type(self):
        """Test error on invalid plugin type."""
        from fitz_ai.llm.schema_defaults import get_all_fields

        with pytest.raises(ValueError, match="Unknown plugin type"):
            get_all_fields("invalid_type")


# =============================================================================
# Chat Plugin Tests
# =============================================================================


class TestChatPlugins:
    """Test chat plugin loading and validation."""

    def test_list_chat_plugins(self):
        """Test listing available chat plugins."""
        from fitz_ai.llm.loader import list_plugins

        plugins = list_plugins("chat")

        assert "openai" in plugins
        assert "cohere" in plugins
        assert "anthropic" in plugins
        assert "local_ollama" in plugins or "ollama" in plugins

    def test_load_openai_plugin(self):
        """Test loading OpenAI chat plugin."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "openai")

        assert spec.plugin_name == "openai"
        assert spec.plugin_type == "chat"
        assert spec.provider.name == "openai"
        assert spec.provider.base_url == "https://api.openai.com/v1"
        assert spec.endpoint.path == "/chat/completions"
        assert spec.request.messages_transform.value == "openai_chat"
        assert "OPENAI_API_KEY" in spec.auth.env_vars

    def test_load_cohere_plugin(self):
        """Test loading Cohere chat plugin."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "cohere")

        assert spec.plugin_name == "cohere"
        assert spec.provider.base_url == "https://api.cohere.ai/v2"
        assert spec.endpoint.path == "/chat"
        assert spec.request.messages_transform.value == "cohere_chat"
        assert "COHERE_API_KEY" in spec.auth.env_vars

    def test_load_anthropic_plugin(self):
        """Test loading Anthropic chat plugin."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "anthropic")

        assert spec.plugin_name == "anthropic"
        assert spec.provider.base_url == "https://api.anthropic.com/v1"
        assert spec.endpoint.path == "/messages"
        assert spec.request.messages_transform.value == "anthropic_chat"
        assert "ANTHROPIC_API_KEY" in spec.auth.env_vars

    def test_load_ollama_plugin(self):
        """Test loading Ollama chat plugin."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "local_ollama")

        assert spec.plugin_name == "ollama"
        assert spec.provider.base_url == "http://localhost:11434"
        assert spec.endpoint.path == "/api/chat"
        assert spec.request.messages_transform.value == "ollama_chat"
        assert spec.auth.type.value == "none"
        assert spec.health_check is not None

    def test_load_azure_openai_plugin(self):
        """Test loading Azure OpenAI chat plugin."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "azure_openai")

        assert spec.plugin_name == "azure_openai"
        assert "{endpoint}" in spec.provider.base_url
        assert len(spec.required_env) == 2

    def test_chat_plugin_structure_consistency(self):
        """Test that all chat plugins have consistent structure."""
        from fitz_ai.llm.loader import list_plugins, load_plugin

        plugins = list_plugins("chat")

        for plugin_name in plugins:
            spec = load_plugin("chat", plugin_name)

            # All must have these fields
            assert spec.plugin_name
            assert spec.plugin_type == "chat"
            assert spec.provider.name
            assert spec.provider.base_url
            assert spec.endpoint.path
            assert spec.endpoint.method
            assert spec.endpoint.timeout > 0
            assert spec.request.messages_transform
            assert spec.response.content_path

    def test_plugin_not_found(self):
        """Test error when plugin doesn't exist."""
        from fitz_ai.llm.loader import YAMLPluginNotFoundError, load_plugin

        with pytest.raises(YAMLPluginNotFoundError):
            load_plugin("chat", "nonexistent_plugin")


# =============================================================================
# Embedding Plugin Tests
# =============================================================================


class TestEmbeddingPlugins:
    """Test embedding plugin loading and validation."""

    def test_list_embedding_plugins(self):
        """Test listing available embedding plugins."""
        from fitz_ai.llm.loader import list_plugins

        plugins = list_plugins("embedding")

        # Should have at least cohere and openai
        assert len(plugins) >= 1

    def test_load_embedding_plugin(self):
        """Test loading an embedding plugin."""
        from fitz_ai.llm.loader import list_plugins, load_plugin

        plugins = list_plugins("embedding")
        if not plugins:
            pytest.skip("No embedding plugins available")

        spec = load_plugin("embedding", plugins[0])

        assert spec.plugin_type == "embedding"
        assert spec.provider.name
        assert spec.provider.base_url
        assert spec.endpoint.path
        assert spec.request.input_field
        assert spec.response.embeddings_path

    def test_embedding_plugin_structure(self):
        """Test embedding plugin has correct structure."""
        from fitz_ai.llm.loader import list_plugins, load_plugin

        plugins = list_plugins("embedding")

        for plugin_name in plugins:
            spec = load_plugin("embedding", plugin_name)

            # Embedding-specific fields
            assert hasattr(spec.request, "input_field")
            assert hasattr(spec.request, "input_wrap")
            assert hasattr(spec.response, "embeddings_path")


# =============================================================================
# Rerank Plugin Tests
# =============================================================================


class TestRerankPlugins:
    """Test rerank plugin loading and validation."""

    def test_list_rerank_plugins(self):
        """Test listing available rerank plugins."""
        from fitz_ai.llm.loader import list_plugins

        plugins = list_plugins("rerank")

        # May or may not have rerank plugins
        assert isinstance(plugins, list)

    def test_load_rerank_plugin(self):
        """Test loading a rerank plugin."""
        from fitz_ai.llm.loader import list_plugins, load_plugin

        plugins = list_plugins("rerank")
        if not plugins:
            pytest.skip("No rerank plugins available")

        spec = load_plugin("rerank", plugins[0])

        assert spec.plugin_type == "rerank"
        assert spec.provider.name
        assert spec.endpoint.path
        assert spec.request.query_field
        assert spec.request.documents_field
        assert spec.response.results_path


# =============================================================================
# Vector DB Plugin Tests (FAISS)
# =============================================================================


class TestFaissVectorDB:
    """Test FAISS vector database plugin."""

    @pytest.fixture
    def temp_faiss_path(self, tmp_path):
        """Create temporary path for FAISS index."""
        return tmp_path / "faiss_test"

    def test_faiss_init(self, temp_faiss_path):
        """Test FAISS initialization."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        assert db.plugin_name == "local-faiss"
        assert db.plugin_type == "vector_db"

    def test_faiss_upsert_and_search(self, temp_faiss_path):
        """Test FAISS upsert and search operations."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        # Create test points
        points = [
            {"id": "doc1", "vector": [1.0, 0.0, 0.0], "payload": {"content": "Hello"}},
            {"id": "doc2", "vector": [0.0, 1.0, 0.0], "payload": {"content": "World"}},
            {"id": "doc3", "vector": [0.0, 0.0, 1.0], "payload": {"content": "Test"}},
        ]

        # Upsert
        db.upsert("test_collection", points)

        # Verify count
        assert db.count("test_collection") == 3

        # Search
        results = db.search("test_collection", [1.0, 0.0, 0.0], limit=2)

        assert len(results) == 2
        assert results[0].id == "doc1"  # Most similar
        assert results[0].score > 0

    def test_faiss_list_collections(self, temp_faiss_path):
        """Test listing collections."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        # Add to multiple collections
        db.upsert("collection_a", [{"id": "a1", "vector": [1.0, 0.0], "payload": {}}])
        db.upsert("collection_b", [{"id": "b1", "vector": [0.0, 1.0], "payload": {}}])

        collections = db.list_collections()

        assert "collection_a" in collections
        assert "collection_b" in collections

    def test_faiss_get_collection_stats(self, temp_faiss_path):
        """Test getting collection statistics."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        db.upsert(
            "stats_test",
            [
                {"id": "s1", "vector": [1.0, 2.0, 3.0], "payload": {}},
                {"id": "s2", "vector": [4.0, 5.0, 6.0], "payload": {}},
            ],
        )

        stats = db.get_collection_stats("stats_test")

        assert stats["points_count"] == 2
        assert stats["vectors_count"] == 2
        assert stats["status"] == "ready"
        assert stats["vector_size"] == 3

    def test_faiss_scroll(self, temp_faiss_path):
        """Test scrolling through records."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        # Add multiple records
        points = [
            {"id": f"scroll_{i}", "vector": [float(i), 0.0], "payload": {"num": i}}
            for i in range(5)
        ]
        db.upsert("scroll_test", points)

        # Scroll with limit
        records, next_offset = db.scroll("scroll_test", limit=2, offset=0)

        assert len(records) == 2
        assert next_offset == 2

        # Continue scrolling
        records2, next_offset2 = db.scroll("scroll_test", limit=2, offset=2)

        assert len(records2) == 2
        assert next_offset2 == 4

        # Last batch
        records3, next_offset3 = db.scroll("scroll_test", limit=2, offset=4)

        assert len(records3) == 1
        assert next_offset3 is None  # No more records

    def test_faiss_delete_collection(self, temp_faiss_path):
        """Test deleting a collection."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        db.upsert("to_delete", [{"id": "d1", "vector": [1.0], "payload": {}}])
        db.upsert("to_keep", [{"id": "k1", "vector": [2.0], "payload": {}}])

        assert db.count("to_delete") == 1
        assert db.count("to_keep") == 1

        deleted = db.delete_collection("to_delete")

        assert deleted == 1
        assert db.count("to_delete") == 0
        assert db.count("to_keep") == 1

    def test_faiss_persistence(self, temp_faiss_path):
        """Test that data persists across instances."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        # Create and populate
        db1 = FaissLocalVectorDB(path=temp_faiss_path)
        db1.upsert(
            "persist_test",
            [{"id": "p1", "vector": [1.0, 2.0], "payload": {"data": "test"}}],
        )

        # Create new instance pointing to same path
        db2 = FaissLocalVectorDB(path=temp_faiss_path)

        # Should find the data
        assert db2.count("persist_test") == 1

        results = db2.search("persist_test", [1.0, 2.0], limit=1)
        assert len(results) == 1
        assert results[0].id == "p1"
        assert results[0].payload.get("data") == "test"

    def test_faiss_empty_search(self, temp_faiss_path):
        """Test searching empty collection."""
        from fitz_ai.backends.local_vector_db.faiss import FaissLocalVectorDB

        db = FaissLocalVectorDB(path=temp_faiss_path)

        results = db.search("nonexistent", [1.0, 0.0], limit=10)

        assert results == []


# =============================================================================
# Integration Tests
# =============================================================================


class TestPluginIntegration:
    """Integration tests for the plugin system."""

    def test_loader_applies_defaults(self):
        """Test that loader applies defaults from schema."""
        from fitz_ai.llm.loader import load_plugin

        spec = load_plugin("chat", "openai")

        # These should have defaults even if not in YAML
        assert spec.version == "1.0"
        assert not spec.response.is_array
        assert spec.response.array_index == 0

    def test_cache_functionality(self):
        """Test that plugin loading is cached."""
        from fitz_ai.llm.loader import clear_cache, load_plugin

        clear_cache()

        # Load twice
        spec1 = load_plugin("chat", "openai")
        spec2 = load_plugin("chat", "openai")

        # Should be same object (cached)
        assert spec1 is spec2

    def test_all_chat_plugins_valid(self):
        """Test that all chat plugins pass validation."""
        from fitz_ai.llm.loader import list_plugins, load_plugin

        plugins = list_plugins("chat")

        for plugin_name in plugins:
            # Should not raise
            spec = load_plugin("chat", plugin_name)
            assert spec.plugin_name == plugin_name or spec.plugin_name in plugin_name


# =============================================================================
# Run tests directly
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

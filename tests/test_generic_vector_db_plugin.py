# fitz_ai/vector_db/tests/test_generic_vector_db_plugin.py
"""
Tests for the Generic HTTP Vector DB Plugin System.

These tests validate:
1. YAML spec loading and parsing
2. GenericVectorDBPlugin operations
3. Point transformation (standard -> provider-specific)
4. UUID conversion for providers that require it
5. Auto-collection creation
6. All YAML plugins load correctly

Run with: pytest fitz_ai/vector_db/tests/test_generic_vector_db_plugin.py -v
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import pytest

from fitz_ai.vector_db.base import SearchResult
from fitz_ai.vector_db.loader import (
    GenericVectorDBPlugin,
    VectorDBSpec,
    _string_to_uuid,
    create_vector_db_plugin,
    load_vector_db_spec,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def qdrant_spec() -> VectorDBSpec:
    """Load the Qdrant YAML spec."""
    return load_vector_db_spec("qdrant")


@pytest.fixture
def pinecone_spec() -> VectorDBSpec:
    """Load the Pinecone YAML spec."""
    return load_vector_db_spec("pinecone")


@pytest.fixture
def weaviate_spec() -> VectorDBSpec:
    """Load the Weaviate YAML spec."""
    return load_vector_db_spec("weaviate")


@pytest.fixture
def milvus_spec() -> VectorDBSpec:
    """Load the Milvus YAML spec."""
    return load_vector_db_spec("milvus")


@pytest.fixture
def sample_points() -> List[Dict[str, Any]]:
    """Sample points in standard format."""
    return [
        {
            "id": "doc1:0",
            "vector": [0.1, 0.2, 0.3, 0.4],
            "payload": {"text": "Hello world", "source": "test.txt"},
        },
        {
            "id": "doc1:1",
            "vector": [0.5, 0.6, 0.7, 0.8],
            "payload": {"text": "Goodbye world", "source": "test.txt"},
        },
    ]


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"result": {"points": []}}
    mock_response.raise_for_status = MagicMock()
    mock_client.request.return_value = mock_response
    mock_client.get.return_value = mock_response
    return mock_client


# =============================================================================
# VectorDBSpec Tests
# =============================================================================


class TestVectorDBSpec:
    """Tests for VectorDBSpec YAML parsing."""

    def test_load_qdrant_spec(self, qdrant_spec: VectorDBSpec):
        """Qdrant spec loads correctly."""
        assert qdrant_spec.name == "qdrant"
        assert qdrant_spec.type == "vector_db"
        assert qdrant_spec.requires_uuid_ids() is True
        assert qdrant_spec.is_local() is False
        assert qdrant_spec.get_auto_detect_service() == "qdrant"

    def test_load_pinecone_spec(self, pinecone_spec: VectorDBSpec):
        """Pinecone spec loads correctly."""
        assert pinecone_spec.name == "pinecone"
        assert pinecone_spec.requires_uuid_ids() is False
        assert pinecone_spec.supports_namespaces() is True
        assert pinecone_spec.get_auto_detect_service() is None

    def test_load_weaviate_spec(self, weaviate_spec: VectorDBSpec):
        """Weaviate spec loads correctly."""
        assert weaviate_spec.name == "weaviate"
        assert weaviate_spec.requires_uuid_ids() is False
        assert weaviate_spec.get_auto_detect_service() == "weaviate"

    def test_load_milvus_spec(self, milvus_spec: VectorDBSpec):
        """Milvus spec loads correctly."""
        assert milvus_spec.name == "milvus"
        assert milvus_spec.requires_uuid_ids() is False

    def test_all_plugins_have_required_operations(self):
        """All HTTP plugins must have search and upsert operations."""
        plugins_dir = Path(__file__).parent.parent / "plugins"

        for yaml_file in plugins_dir.glob("*.yaml"):
            spec = VectorDBSpec(yaml_file)

            # Skip local plugins
            if spec.is_local():
                continue

            assert "search" in spec.operations, f"{spec.name} missing search"
            assert "upsert" in spec.operations, f"{spec.name} missing upsert"

    def test_build_base_url_qdrant(self, qdrant_spec: VectorDBSpec):
        """Qdrant base URL builds correctly."""
        url = qdrant_spec.build_base_url(host="myhost", port=1234)
        assert url == "http://myhost:1234"

    def test_build_base_url_qdrant_defaults(self, qdrant_spec: VectorDBSpec):
        """Qdrant uses default host/port when not provided."""
        url = qdrant_spec.build_base_url()
        assert url == "http://localhost:6333"

    def test_build_base_url_pinecone(self, pinecone_spec: VectorDBSpec):
        """Pinecone base URL builds correctly with all params."""
        url = pinecone_spec.build_base_url(
            index_name="my-index",
            project_id="abc123",
            environment="us-west1-gcp",
        )
        assert url == "https://my-index-abc123.svc.us-west1-gcp.pinecone.io"

    def test_render_template_direct_substitution(self, qdrant_spec: VectorDBSpec):
        """Direct variable substitution returns actual type, not string."""
        template = "{{query_vector}}"
        context = {"query_vector": [0.1, 0.2, 0.3]}
        result = qdrant_spec.render_template(template, context)
        assert result == [0.1, 0.2, 0.3]
        assert isinstance(result, list)

    def test_render_template_nested(self, qdrant_spec: VectorDBSpec):
        """Nested templates render correctly."""
        template = {
            "vector": "{{query_vector}}",
            "limit": "{{limit}}",
            "nested": {"value": "{{with_payload}}"},
        }
        context = {"query_vector": [0.1], "limit": 10, "with_payload": True}
        result = qdrant_spec.render_template(template, context)

        assert result["vector"] == [0.1]
        assert result["limit"] == 10
        assert result["nested"]["value"] is True


# =============================================================================
# UUID Conversion Tests
# =============================================================================


class TestUUIDConversion:
    """Tests for UUID conversion logic."""

    def test_string_to_uuid_deterministic(self):
        """Same string always produces same UUID."""
        id1 = _string_to_uuid("doc.txt:0")
        id2 = _string_to_uuid("doc.txt:0")
        assert id1 == id2

    def test_string_to_uuid_different_strings(self):
        """Different strings produce different UUIDs."""
        id1 = _string_to_uuid("doc.txt:0")
        id2 = _string_to_uuid("doc.txt:1")
        assert id1 != id2

    def test_string_to_uuid_valid_format(self):
        """Generated UUID is valid."""
        result = _string_to_uuid("test")
        parsed = uuid.UUID(result)
        assert str(parsed) == result


# =============================================================================
# Point Transformation Tests
# =============================================================================


class TestPointTransformation:
    """Tests for point format transformation."""

    def test_qdrant_identity_transform(
        self, qdrant_spec: VectorDBSpec, sample_points: List[Dict]
    ):
        """Qdrant uses identity transform (no changes)."""
        transformed = qdrant_spec.transform_points(sample_points, "upsert")
        assert transformed == sample_points

    def test_pinecone_transform(
        self, pinecone_spec: VectorDBSpec, sample_points: List[Dict]
    ):
        """Pinecone transforms vector->values, payload->metadata."""
        transformed = pinecone_spec.transform_points(sample_points, "upsert")

        assert len(transformed) == 2
        assert "values" in transformed[0]
        assert "metadata" in transformed[0]
        assert transformed[0]["values"] == [0.1, 0.2, 0.3, 0.4]
        assert transformed[0]["metadata"] == sample_points[0]["payload"]


# =============================================================================
# GenericVectorDBPlugin Tests
# =============================================================================


class TestGenericVectorDBPlugin:
    """Tests for GenericVectorDBPlugin operations."""

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_search_returns_search_results(
        self, mock_client_class, qdrant_spec: VectorDBSpec
    ):
        """Search returns properly formatted SearchResult objects."""
        # Setup mock
        mock_response = MagicMock()
        mock_response.status_code = 200
        # FIX: Qdrant 1.0+ /points/query returns {"result": {"points": [...]}}
        mock_response.json.return_value = {
            "result": {
                "points": [
                    {"id": "abc-123", "score": 0.95, "payload": {"text": "hello"}},
                    {"id": "def-456", "score": 0.87, "payload": {"text": "world"}},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Create plugin and search
        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        results = plugin.search("test_collection", [0.1, 0.2, 0.3], limit=10)

        # Verify
        assert len(results) == 2
        assert isinstance(results[0], SearchResult)
        assert results[0].id == "abc-123"
        assert results[0].score == 0.95
        assert results[0].payload == {"text": "hello"}

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_upsert_converts_ids_for_qdrant(
        self, mock_client_class, qdrant_spec: VectorDBSpec, sample_points: List[Dict]
    ):
        """Qdrant upsert converts string IDs to UUIDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        plugin.upsert("test_collection", sample_points)

        # Verify the request was made
        mock_client.request.assert_called()
        call_kwargs = mock_client.request.call_args

        # Check that points were sent in the body
        body = call_kwargs.kwargs.get("json", {})
        points = body.get("points", [])

        # IDs should be converted to UUIDs
        assert len(points) == 2
        for point in points:
            # Should be a valid UUID
            try:
                uuid.UUID(point["id"])
            except ValueError:
                pytest.fail(f"ID {point['id']} is not a valid UUID")

            # Original ID should be preserved in payload
            assert "_original_id" in point["payload"]

    @patch.dict(os.environ, {"PINECONE_API_KEY": "test-api-key"})
    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_upsert_no_uuid_conversion_for_pinecone(
        self, mock_client_class, pinecone_spec: VectorDBSpec, sample_points: List[Dict]
    ):
        """Pinecone upsert keeps original string IDs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(
            pinecone_spec,
            index_name="test",
            project_id="abc",
            environment="us-east-1-aws",
        )
        plugin.upsert("test_namespace", sample_points)

        # Verify
        call_kwargs = mock_client.request.call_args
        body = call_kwargs.kwargs.get("json", {})
        vectors = body.get("vectors", [])

        # IDs should remain as strings (not UUIDs)
        assert vectors[0]["id"] == "doc1:0"

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_count_returns_integer(self, mock_client_class, qdrant_spec: VectorDBSpec):
        """Count returns an integer."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {"points_count": 42, "status": "green"}
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        count = plugin.count("test_collection")

        assert count == 42
        assert isinstance(count, int)

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_list_collections(self, mock_client_class, qdrant_spec: VectorDBSpec):
        """List collections returns list of names."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "collections": [
                    {"name": "collection1"},
                    {"name": "collection2"},
                ]
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        collections = plugin.list_collections()

        assert collections == ["collection1", "collection2"]

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_auto_create_collection_on_404(
        self, mock_client_class, qdrant_spec: VectorDBSpec, sample_points: List[Dict]
    ):
        """Auto-creates collection when upsert gets 404."""
        # First call returns 404, second succeeds
        mock_response_404 = MagicMock()
        mock_response_404.status_code = 404
        mock_response_404.raise_for_status.side_effect = Exception("404")

        mock_response_ok = MagicMock()
        mock_response_ok.status_code = 200
        mock_response_ok.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.side_effect = [
            mock_response_404,  # First upsert fails
            mock_response_ok,  # Create collection succeeds
            mock_response_ok,  # Retry upsert succeeds
        ]
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        plugin.upsert("new_collection", sample_points)

        # Should have made 3 calls: upsert, create, upsert
        assert mock_client.request.call_count == 3

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_delete_collection(self, mock_client_class, qdrant_spec: VectorDBSpec):
        """Delete collection works correctly."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        plugin.delete_collection("test_collection")

        mock_client.request.assert_called_once()
        call_args = mock_client.request.call_args
        assert call_args.kwargs["method"] == "DELETE"
        assert "test_collection" in call_args.kwargs["url"]

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_get_collection_stats(self, mock_client_class, qdrant_spec: VectorDBSpec):
        """Get collection stats returns dict."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "result": {
                "points_count": 100,
                "segments_count": 2,
                "status": "green",
            }
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.get.return_value = mock_response
        mock_client_class.return_value = mock_client

        plugin = GenericVectorDBPlugin(qdrant_spec, host="localhost", port=6333)
        stats = plugin.get_collection_stats("test_collection")

        assert stats["points_count"] == 100
        assert stats["status"] == "green"


# =============================================================================
# Plugin Factory Tests
# =============================================================================


class TestCreateVectorDBPlugin:
    """Tests for create_vector_db_plugin factory function."""

    def test_unknown_plugin_raises(self):
        """Unknown plugin name raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            create_vector_db_plugin("nonexistent_db")

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_creates_generic_plugin_for_http(self, mock_client_class):
        """HTTP plugins return GenericVectorDBPlugin."""
        mock_client_class.return_value = MagicMock()

        plugin = create_vector_db_plugin("qdrant", host="localhost", port=6333)

        assert isinstance(plugin, GenericVectorDBPlugin)
        assert plugin.plugin_name == "qdrant"

    def test_local_plugin_loads_python_class(self):
        """Local plugins load the specified Python class."""
        # Check if the local_faiss.yaml file exists
        plugins_dir = Path(__file__).parent.parent / "plugins"
        local_faiss_yaml = plugins_dir / "local_faiss.yaml"

        if not local_faiss_yaml.exists():
            pytest.skip("local_faiss.yaml not found in plugins directory")

        # This will fail if faiss is not installed, which is expected
        # The test validates that the loader attempts to load the class
        try:
            plugin = create_vector_db_plugin("local_faiss")
            assert (
                plugin.plugin_name == "local_faiss"
                or plugin.plugin_name == "local-faiss"
            )
        except ImportError:
            # faiss not installed - that's fine, we tested the path
            pytest.skip("faiss not installed")
        except ValueError as e:
            if "not found" in str(e):
                pytest.skip("local_faiss plugin not installed")
            raise


# =============================================================================
# Integration-style Tests (with mocked HTTP)
# =============================================================================


class TestFullWorkflow:
    """End-to-end workflow tests with mocked HTTP."""

    @patch("fitz_ai.vector_db.loader.httpx.Client")
    def test_ingest_and_search_workflow(self, mock_client_class):
        """Test complete ingest -> search workflow."""
        # Setup mock responses
        upsert_response = MagicMock()
        upsert_response.status_code = 200
        upsert_response.raise_for_status = MagicMock()

        search_response = MagicMock()
        search_response.status_code = 200
        # FIX: Qdrant 1.0+ /points/query returns {"result": {"points": [...]}}
        search_response.json.return_value = {
            "result": {
                "points": [
                    {"id": "doc1", "score": 0.95, "payload": {"text": "result 1"}},
                ]
            }
        }
        search_response.raise_for_status = MagicMock()

        mock_client = MagicMock()
        mock_client.request.side_effect = [upsert_response, search_response]
        mock_client_class.return_value = mock_client

        # Create plugin
        db = create_vector_db_plugin("qdrant", host="localhost", port=6333)

        # Upsert
        points = [
            {"id": "doc1", "vector": [0.1, 0.2], "payload": {"text": "test"}},
        ]
        db.upsert("my_collection", points)

        # Search
        results = db.search("my_collection", [0.1, 0.2], limit=5)

        assert len(results) == 1
        assert results[0].score == 0.95


# =============================================================================
# YAML Schema Validation Tests
# =============================================================================


class TestYAMLSchemaCompliance:
    """Tests that all YAML plugins follow the schema."""

    def test_all_plugins_have_name(self):
        """All plugins have a name field."""
        plugins_dir = Path(__file__).parent.parent / "plugins"

        for yaml_file in plugins_dir.glob("*.yaml"):
            spec = VectorDBSpec(yaml_file)
            assert spec.name, f"{yaml_file.name} missing name"

    def test_all_plugins_have_type(self):
        """All plugins have type: vector_db."""
        plugins_dir = Path(__file__).parent.parent / "plugins"

        for yaml_file in plugins_dir.glob("*.yaml"):
            spec = VectorDBSpec(yaml_file)
            assert spec.type == "vector_db", f"{yaml_file.name} wrong type"

    def test_http_plugins_have_connection(self):
        """HTTP plugins have connection config."""
        plugins_dir = Path(__file__).parent.parent / "plugins"

        for yaml_file in plugins_dir.glob("*.yaml"):
            spec = VectorDBSpec(yaml_file)
            assert spec.connection, f"{yaml_file.name} missing connection"

    def test_search_operation_has_response_mapping(self):
        """Search operations have response mapping."""
        plugins_dir = Path(__file__).parent.parent / "plugins"

        for yaml_file in plugins_dir.glob("*.yaml"):
            spec = VectorDBSpec(yaml_file)

            if spec.is_local():
                continue

            search_op = spec.operations.get("search", {})
            response = search_op.get("response", {})

            assert "results_path" in response, (
                f"{spec.name} search missing results_path"
            )
            assert "mapping" in response, f"{spec.name} search missing mapping"


# =============================================================================
# Auth Tests
# =============================================================================


class TestAuthentication:
    """Tests for authentication handling."""

    def test_optional_auth_without_env_var(self, qdrant_spec: VectorDBSpec):
        """Optional auth skips if env var not set."""
        # Clear env var if it exists
        env_backup = os.environ.pop("QDRANT_API_KEY", None)
        try:
            # Qdrant has optional auth
            headers = qdrant_spec.get_auth_headers()
            # Should return empty dict since QDRANT_API_KEY is not set
            assert headers == {}
        finally:
            if env_backup:
                os.environ["QDRANT_API_KEY"] = env_backup

    @patch.dict(os.environ, {"QDRANT_API_KEY": "my-secret-key"})
    def test_optional_auth_with_env_var(self, qdrant_spec: VectorDBSpec):
        """Optional auth includes header when env var is set."""
        headers = qdrant_spec.get_auth_headers()
        assert "api-key" in headers
        assert headers["api-key"] == "my-secret-key"

    def test_required_auth_raises_without_env_var(self, pinecone_spec: VectorDBSpec):
        """Required auth raises ValueError if env var not set."""
        # Clear the env var if it exists
        env_backup = os.environ.pop("PINECONE_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="PINECONE_API_KEY not set"):
                pinecone_spec.get_auth_headers()
        finally:
            if env_backup:
                os.environ["PINECONE_API_KEY"] = env_backup

    @patch.dict(os.environ, {"PINECONE_API_KEY": "pine-key-123"})
    def test_required_auth_with_env_var(self, pinecone_spec: VectorDBSpec):
        """Required auth includes header when env var is set."""
        headers = pinecone_spec.get_auth_headers()
        assert "Api-Key" in headers
        assert headers["Api-Key"] == "pine-key-123"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

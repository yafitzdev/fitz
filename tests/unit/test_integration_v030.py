# tests/unit/test_integration_v030.py
"""
Integration tests for core architecture.

These tests verify that the complete refactored system works end-to-end:
- Core contracts are usable
- Engine wrapper works
- Runtime works
- Registry works
- Universal runner works
"""

from unittest.mock import Mock

import pytest

# Core imports
from fitz_sage.core import (
    Answer,
    EnrichmentError,
    GenerationError,
    KnowledgeEngine,
    KnowledgeError,
    ManagedModelError,
    Provenance,
    Query,
    QueryError,
    QueryIntelligenceError,
)

# Runtime imports
from fitz_sage.runtime import (
    EngineRegistry,
    create_engine,
    get_engine_registry,
    list_engines,
    list_engines_with_info,
    run,
)


class TestCoreContracts:
    """Test that core contracts work as expected."""

    def test_query_creation(self):
        """Test Query object creation and validation."""
        # Simple query
        query = Query(text="What is quantum computing?")
        assert query.text == "What is quantum computing?"
        assert query.metadata == {}

        # Query with metadata
        query = Query(text="Test", metadata={"temp": 0.3})
        assert query.metadata["temp"] == 0.3

    def test_query_validation(self):
        """Test that empty queries are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="")

        with pytest.raises(ValueError, match="cannot be empty"):
            Query(text="   ")

    def test_answer_creation(self):
        """Test Answer object creation."""
        # Simple answer
        answer = Answer(text="Quantum computing uses qubits")
        assert answer.text == "Quantum computing uses qubits"
        assert answer.provenance == []
        assert answer.metadata == {}

        # Answer with provenance
        prov = Provenance(source_id="doc_1", excerpt="Qubits...")
        answer = Answer(text="Answer", provenance=[prov])
        assert len(answer.provenance) == 1
        assert answer.provenance[0].source_id == "doc_1"

    def test_provenance_creation(self):
        """Test Provenance object creation."""
        prov = Provenance(source_id="doc_123", excerpt="Some text", metadata={"title": "Paper"})
        assert prov.source_id == "doc_123"
        assert prov.excerpt == "Some text"
        assert prov.metadata["title"] == "Paper"


class TestEngineRegistry:
    """Test the engine registry."""

    def setup_method(self):
        """Reset registry before each test."""
        self._saved_registry = EngineRegistry._global_registry
        EngineRegistry.reset_global()

    def teardown_method(self):
        """Restore global registry after each test."""
        EngineRegistry._global_registry = self._saved_registry

    def test_singleton_registry(self):
        """Test that global registry is a singleton."""
        reg1 = get_engine_registry()
        reg2 = get_engine_registry()
        assert reg1 is reg2

    def test_register_engine(self):
        """Test engine registration."""
        registry = get_engine_registry()

        def mock_factory(config):
            return Mock()

        registry.register(name="test_engine", factory=mock_factory, description="Test engine")

        assert "test_engine" in registry.list()

    def test_register_duplicate_raises(self):
        """Test that registering duplicate raises error."""
        registry = get_engine_registry()

        def mock_factory(config):
            return Mock()

        registry.register("test", mock_factory, "Test")

        with pytest.raises(ValueError, match="already registered"):
            registry.register("test", mock_factory, "Test 2")

    def test_get_engine_factory(self):
        """Test retrieving engine factory."""
        registry = get_engine_registry()

        def mock_factory(config):
            return Mock(spec=KnowledgeEngine)

        registry.register("test", mock_factory)

        factory = registry.get("test")
        # Factory may be wrapped for instrumentation, so check it's callable
        assert callable(factory)

        engine = factory(None)
        # Engine may be wrapped in InstrumentedProxy, check underlying behavior
        assert hasattr(engine, "answer")

    def test_get_unknown_engine_raises(self):
        """Test that getting unknown engine raises error."""
        registry = get_engine_registry()

        with pytest.raises(Exception, match="Unknown engine"):
            registry.get("nonexistent")

    def test_list_engines(self):
        """Test listing engines."""
        registry = get_engine_registry()

        def mock_factory(config):
            return Mock()

        registry.register("engine1", mock_factory)
        registry.register("engine2", mock_factory)

        engines = registry.list()
        assert "engine1" in engines
        assert "engine2" in engines

    def test_list_with_descriptions(self):
        """Test listing engines with descriptions."""
        registry = get_engine_registry()

        def mock_factory(config):
            return Mock()

        registry.register("test1", mock_factory, "First engine")
        registry.register("test2", mock_factory, "Second engine")

        info = registry.list_with_descriptions()
        assert info["test1"] == "First engine"
        assert info["test2"] == "Second engine"

    def test_decorator_registration(self):
        """Test decorator-based registration."""
        registry = get_engine_registry()

        @EngineRegistry.register_engine(name="decorated", description="Decorated engine")
        def create_engine(config):
            return Mock(spec=KnowledgeEngine)

        assert "decorated" in registry.list()
        factory = registry.get("decorated")
        # Factory may be wrapped for instrumentation, so check it's callable
        assert callable(factory)
        # Verify the factory produces engines
        engine = factory(None)
        assert hasattr(engine, "answer")


class TestUniversalRunner:
    """Test the universal runner."""

    def setup_method(self):
        """Reset registry and set up mock engine."""
        self._saved_registry = EngineRegistry._global_registry
        EngineRegistry.reset_global()

        # Register a mock engine
        registry = get_engine_registry()

        def mock_factory(config):
            mock_engine = Mock(spec=KnowledgeEngine)
            mock_engine.answer.return_value = Answer(
                text="Mock answer",
                provenance=[Provenance(source_id="mock_1")],
                metadata={"engine": "mock"},
            )
            return mock_engine

        registry.register(
            name="mock_engine",
            factory=mock_factory,
            description="Mock engine for testing",
        )

    def teardown_method(self):
        """Restore global registry after each test."""
        EngineRegistry._global_registry = self._saved_registry

    def test_list_engines_function(self):
        """Test list_engines convenience function."""
        engines = list_engines()
        assert "mock_engine" in engines

    def test_list_engines_with_info_function(self):
        """Test list_engines_with_info function."""
        info = list_engines_with_info()
        assert "mock_engine" in info
        assert info["mock_engine"] == "Mock engine for testing"

    def test_create_engine_function(self):
        """Test create_engine function."""
        engine = create_engine(engine="mock_engine", config=None)
        assert hasattr(engine, "answer")

    def test_run_with_string_query(self):
        """Test universal run() with string query."""
        answer = run(query="What is X?", engine="mock_engine", config=None)

        assert answer.text == "Mock answer"
        assert len(answer.provenance) == 1
        assert answer.provenance[0].source_id == "mock_1"

    def test_run_with_query_object(self):
        """Test universal run() with Query object."""
        query = Query(text="What is Y?")
        answer = run(query=query, engine="mock_engine", config=None)

        assert answer.text == "Mock answer"

    def test_run_unknown_engine_raises(self):
        """Test that unknown engine raises error."""
        with pytest.raises(Exception, match="Unknown engine"):
            run(query="What is X?", engine="nonexistent", config=None)


class TestProtocolCompliance:
    """Test that engines comply with KnowledgeEngine protocol."""

    def test_protocol_check(self):
        """Test that protocol checking works."""

        # Mock engine that implements protocol
        class GoodEngine:
            def answer(self, query: Query) -> Answer:
                return Answer(text="Test")

        # Mock engine that doesn't implement protocol
        class BadEngine:
            def query(self, q: str) -> str:
                return "Test"

        # Check protocol compliance
        good = GoodEngine()
        bad = BadEngine()

        assert isinstance(good, KnowledgeEngine)
        assert not isinstance(bad, KnowledgeEngine)


class TestErrorHandling:
    """Test error handling across the system."""

    def test_query_error_hierarchy(self):
        """Test exception hierarchy."""
        from fitz_sage.core import EngineError

        assert issubclass(QueryError, EngineError)
        assert issubclass(KnowledgeError, EngineError)
        assert issubclass(GenerationError, EngineError)
        assert issubclass(QueryIntelligenceError, GenerationError)
        assert issubclass(EnrichmentError, KnowledgeError)
        assert issubclass(ManagedModelError, EngineError)

    def test_query_error_can_be_raised(self):
        """Test that errors can be raised and caught."""
        with pytest.raises(QueryError):
            raise QueryError("Test error")

        # Test catching as EngineError
        from fitz_sage.core import EngineError

        with pytest.raises(EngineError):
            raise QueryError("Test error")


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])

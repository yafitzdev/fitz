# tests/unit/test_krag_registration.py
"""Tests for fitz_krag engine registration."""


class TestKragRegistration:
    def test_engine_registers_in_global_registry(self):
        # Importing runtime triggers auto-registration
        import fitz_sage.engines.fitz_krag.runtime  # noqa: F401
        from fitz_sage.runtime.registry import EngineRegistry

        registry = EngineRegistry.get_global()
        assert "fitz_krag" in registry.list()

    def test_engine_capabilities(self):
        import fitz_sage.engines.fitz_krag.runtime  # noqa: F401
        from fitz_sage.runtime.registry import EngineRegistry

        registry = EngineRegistry.get_global()
        caps = registry.get_capabilities("fitz_krag")
        assert caps.supports_collections is True
        assert caps.supports_persistent_ingest is True
        assert caps.supports_chat is True
        assert caps.requires_config is True

    def test_config_loader_registered(self):
        import fitz_sage.engines.fitz_krag.runtime  # noqa: F401
        from fitz_sage.runtime.registry import EngineRegistry

        registry = EngineRegistry.get_global()
        info = registry.get_info("fitz_krag")
        assert info.config_loader is not None
        assert info.config_type is not None

    def test_default_config_path_registered(self):
        import fitz_sage.engines.fitz_krag.runtime  # noqa: F401
        from fitz_sage.runtime.registry import EngineRegistry

        registry = EngineRegistry.get_global()
        path = registry.get_default_config_path("fitz_krag")
        assert path is not None
        assert path.exists()

    def test_config_loader_loads_defaults(self):
        from unittest.mock import patch

        from fitz_sage.config.loader import load_engine_config
        from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig

        with patch("fitz_sage.config.loader._load_user_config", return_value=None):
            config = load_engine_config("fitz_krag")
        assert isinstance(config, FitzKragConfig)
        assert config.collection  # has a collection value
        assert config.chat_smart is None  # retrieval default has no chat provider

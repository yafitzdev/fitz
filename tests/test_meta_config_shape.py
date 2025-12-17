from fitz.engines.classic_rag.config.loader import DEFAULT_CONFIG_PATH, _load_yaml


def test_default_yaml_is_meta_config():
    data = _load_yaml(DEFAULT_CONFIG_PATH)

    assert "default_preset" in data
    assert "presets" in data
    assert isinstance(data["presets"], dict)

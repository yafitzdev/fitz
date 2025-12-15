# tests/test_preset_integration.py
from fitz.core.config.presets import get_preset
from fitz.pipeline.pipeline.engine import RAGPipeline


def test_preset_works_with_pipeline():
    """Test that presets work with RAGPipeline."""
    config = get_preset("local")

    # Should be able to create pipeline from preset
    pipeline = RAGPipeline.from_dict(config)

    # Should be able to run queries
    answer = pipeline.run("test query")
    assert answer is not None
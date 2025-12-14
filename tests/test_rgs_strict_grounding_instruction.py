# tests/test_rgs_strict_grounding_instruction.py
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def test_rgs_strict_grounding_instruction_present():
    rgs = RGS(RGSConfig(strict_grounding=True))

    chunks = [{"id": "1", "content": "alpha", "metadata": {}}]
    prompt = rgs.build_prompt("Q?", chunks)

    assert "I don't know based on the provided information." in prompt.system

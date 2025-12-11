# tests/test_rgs_strict_grounding_instruction.py

from fitz_rag.generation.rgs import RGS, RGSConfig

def test_rgs_strict_grounding_instruction_present():
    cfg = RGSConfig(strict_grounding=True)
    rgs = RGS(cfg)

    chunks = [{"id": "1", "text": "alpha", "metadata": {}}]
    prompt = rgs.build_prompt("Q?", chunks)

    assert "I don't know based on the provided information" in prompt.system

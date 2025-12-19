# tests/test_rgs_prompt_slots.py
from fitz_ai.engines.classic_rag.generation.prompting import PromptConfig
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSConfig


def test_rgs_prompt_slots_override_defaults():
    cfg = RGSConfig(
        prompt_config=PromptConfig(
            system_base="SYSTEM BASE OVERRIDE",
            user_instructions="Do the thing in bullet points.",
        )
    )
    rgs = RGS(cfg)

    chunks = [{"id": "1", "content": "alpha", "metadata": {"file": "doc1"}}]
    prompt = rgs.build_prompt("Q?", chunks)

    assert "SYSTEM BASE OVERRIDE" in prompt.system
    assert "You are a retrieval-grounded assistant." not in prompt.system

    assert "Do the thing in bullet points." in prompt.user

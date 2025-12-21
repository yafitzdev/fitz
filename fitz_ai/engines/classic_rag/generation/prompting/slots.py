# pipeline/generation/prompting/slots.py
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PromptSlots:
    """
    Default prompt slot texts (language only).
    Assembly logic lives elsewhere.
    """

    system_base: str = "You are a retrieval-grounded assistant."
    system_grounding: str = "You must answer ONLY using the provided context snippets."
    system_safety: str = 'If the answer is not contained in the context, say "I don\'t know based on the provided information."'

    context_header: str = "You are given the following context snippets:"
    context_item: str = "{header}\n{content}\n"

    user_instructions: str = ""

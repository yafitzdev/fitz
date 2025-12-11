from __future__ import annotations

from typing import Protocol, List

from fitz_rag.core import Chunk  # noqa: F401  # reserved for future richer APIs


class EmbeddingPlugin(Protocol):
    """
    Protocol for embedding plugins.

    Any embedding implementation (Cohere, OpenAI, local, etc.) should
    implement this interface.

    Plugins typically live in:
        fitz_rag.llm.embedding.plugins.<name>

    and declare a unique `plugin_name` attribute (string).
    """

    # Each plugin should define a class attribute:
    #   plugin_name: str = "<unique-name>"
    #
    # This is used by the auto-discovery registry.

    def embed(self, text: str) -> List[float]:
        """
        Embed a single piece of text into a vector of floats.
        """
        ...

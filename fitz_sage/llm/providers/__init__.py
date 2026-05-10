# fitz_sage/llm/providers/__init__.py
"""
LLM provider implementations.

There is one chat-protocol implementation: ``OpenAICompatChat`` /
``OpenAICompatEmbedding`` / ``OpenAICompatVision`` — an OpenAI HTTP
client that talks to OpenAI itself, Azure OpenAI, llama.cpp,
vLLM, LM Studio, Together, Fireworks, Groq, OpenRouter, and any
other server speaking the protocol.

The ``enterprise`` path is kept separately because its OAuth2 +
API-key composite auth and certificate handling do not fit cleanly
into the simple ``endpoint`` config surface.
"""

from fitz_sage.llm.providers.base import (
    ChatProvider,
    EmbeddingProvider,
    ModelTier,
    RerankProvider,
    RerankResult,
    StreamingChatProvider,
    VisionProvider,
)

__all__ = [
    # Protocols
    "ChatProvider",
    "StreamingChatProvider",
    "EmbeddingProvider",
    "RerankProvider",
    "VisionProvider",
    # Types
    "ModelTier",
    "RerankResult",
]

# OpenAI-compatible HTTP provider (the only chat path).
try:
    from fitz_sage.llm.providers.openai_compat import (  # noqa: F401
        OpenAICompatChat,
        OpenAICompatEmbedding,
        OpenAICompatVision,
    )

    __all__.extend(
        ["OpenAICompatChat", "OpenAICompatEmbedding", "OpenAICompatVision"]
    )
except ImportError:
    # openai SDK not installed — provider class import is optional so
    # static tooling on a fresh checkout still works.
    pass

# LLM-based reranker — the canonical rerank backend after the
# cohere/rerank deletion. Pure Python, no extra deps.
from fitz_sage.llm.providers.llm_reranker import LLMReranker  # noqa: E402

__all__.append("LLMReranker")

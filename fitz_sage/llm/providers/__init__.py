# fitz_sage/llm/providers/__init__.py
"""
LLM provider implementations.

There is one chat-protocol implementation: ``OpenAICompatChat`` /
``OpenAICompatVision`` — an OpenAI HTTP client that talks to OpenAI
itself, Azure OpenAI, llama.cpp, vLLM, LM Studio, Together, Fireworks,
Groq, OpenRouter, and any other server speaking the protocol.

The ``enterprise`` path is kept separately because its OAuth2 +
API-key composite auth and certificate handling do not fit cleanly
into the simple ``endpoint`` config surface.
"""

from fitz_sage.llm.providers.base import (
    ChatProvider,
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
        OpenAICompatVision,
    )

    __all__.extend(["OpenAICompatChat", "OpenAICompatVision"])
except ImportError:
    # openai SDK not installed — provider class import is optional so
    # static tooling on a fresh checkout still works.
    pass

# ONNX cross-encoder reranker — canonical rerank backend (gte-reranker-
# modernbert-base by default). Loads lazily so static tooling on a fresh
# checkout still works without optimum/transformers installed yet.
try:
    from fitz_sage.llm.providers.onnx_reranker import OnnxReranker  # noqa: F401

    __all__.append("OnnxReranker")
except ImportError:
    pass

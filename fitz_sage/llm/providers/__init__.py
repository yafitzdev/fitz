# fitz_sage/llm/providers/__init__.py
"""
LLM provider implementations.

``OnnxChat`` is the managed local Qwen3.5 0.8B enrichment runtime.
``OpenAICompatChat`` / ``OpenAICompatVision`` are optional OpenAI HTTP
clients for user-supplied endpoints such as OpenAI itself, Azure OpenAI,
vLLM, LM Studio, Together, Fireworks, Groq, OpenRouter, and any other
server speaking the protocol.

The ``enterprise`` path is kept separately because its OAuth2 +
API-key composite auth and certificate handling do not fit cleanly
into the simple ``endpoint`` config surface.
"""

from fitz_sage.llm.providers.base import (
    ChatProvider,
    ModelTier,
    RerankProvider,
    RerankResult,
    VisionProvider,
)

__all__ = [
    # Protocols
    "ChatProvider",
    "RerankProvider",
    "VisionProvider",
    # Types
    "ModelTier",
    "RerankResult",
]

# Managed ONNX chat provider for required enrichment.
from fitz_sage.llm.providers.onnx_chat import OnnxChat  # noqa: E402,F401

__all__.append("OnnxChat")

# OpenAI-compatible HTTP provider for optional endpoint/cloud chat paths.
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
# modernbert-base by default). The module imports cleanly with just numpy;
# onnxruntime / transformers / huggingface_hub are imported lazily on the
# first rerank() call.
from fitz_sage.llm.providers.onnx_reranker import OnnxReranker  # noqa: E402,F401

__all__.append("OnnxReranker")

<!-- docs/features/platform/openai-compatible-endpoint.md -->
# OpenAI-Compatible Endpoint Architecture

**Status:** the optional endpoint/cloud chat path. Required enrichment uses the
managed `onnx/qwen3.5-0.8b` provider; endpoint chat is for optional synthesis,
query intelligence, and vision.

## TL;DR

There is one HTTP chat implementation in fitz-sage: `OpenAICompatChat` in
`fitz_sage/llm/providers/openai_compat.py`. It speaks the OpenAI HTTP protocol
against any compliant server.

Provider names are configuration presets on top of that implementation:

| Spec                          | base_url                          | Auth                       | Use case                                            |
|-------------------------------|-----------------------------------|----------------------------|-----------------------------------------------------|
| `endpoint` / `endpoint/<m>`   | required (user-supplied)          | `NoAuth` or `ApiKeyAuth`   | local endpoint / vLLM / LM Studio / any cloud       |
| `openai` / `openai/<m>`       | `https://api.openai.com/v1`       | `OPENAI_API_KEY`           | OpenAI public API                                   |
| `azure_openai/<deployment>`   | required (tenant-specific)        | `AZURE_OPENAI_API_KEY`     | Azure OpenAI                                        |
| `enterprise/<provider>/<m>`   | required                          | OAuth2 + downstream key    | Internal corporate gateway                          |

The legacy `ollama`, `cohere`, `anthropic` provider names have been
removed. Passing them raises an actionable migration error.

## Why a single protocol?

Earlier fitz-sage shipped per-provider integrations (OpenAI, Cohere,
Anthropic, Ollama) as separate code paths. This forced two coupled
choices on every user:

1. Pick a chat provider that *also* had a viable embedding model.
2. Locally, swap models mid-query (Ollama unloads/reloads between chat
   and embedding) or run two SDK clients side by side.

Both were friction. The endpoint architecture removes them by
speaking the OpenAI HTTP protocol — which virtually every modern
serving stack already exposes:

- vLLM, LM Studio, TabbyAPI, Aphrodite, text-generation-webui
- Ollama (`/v1/` mode at `:11434/v1`)
- OpenAI, Together, Fireworks, Groq, DeepInfra, OpenRouter, Mistral La Plateforme

Combined with the v0.12.0 decision to drop the embedding API entirely
(retrieval is BM25 + KRAG routing + ONNX cross-encoder reranker), the chat protocol is
the *only* network protocol fitz-sage speaks.

## Authentication

The auth layer (`fitz_sage.llm.auth`) is shared across all presets:

- `NoAuth` — no headers (default for `endpoint` against an
  unauthenticated local server).
- `ApiKeyAuth` — single header from an env var (`Authorization: Bearer …`).
- `M2MAuth` — OAuth2 client-credentials flow with automatic refresh.
- `CompositeAuth` (used by `enterprise`) — M2M bearer + downstream API key.

For `endpoint`, opt in to `ApiKeyAuth` via:

```yaml
synthesizer: endpoint/meta-llama-3.1-70b-instruct
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY
```

## Engine config (FitzKrag)

```yaml
fitz_krag:
  synthesizer: endpoint/qwen2.5-7b-instruct
  chat_base_url: http://localhost:8080/v1
  chat_api_key_env: null         # unauthenticated local server

  rerank: onnx                   # INT8 ONNX cross-encoder, local on CPU
  collection: default
```

Use role-specific provider fields (`query_intelligence`, `vision`, and
`synthesizer`) to mix local and cloud models. Required `enricher:` and
`summarizer:` default to managed ONNX and do not need an endpoint.

## Cloud quick reference

```yaml
# OpenAI (preset; no base_url needed)
synthesizer: openai/gpt-4o
# OPENAI_API_KEY in env

# Together (endpoint with API key)
synthesizer: endpoint/meta-llama-3.1-70b-instruct
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY

# OpenRouter (gateway over many vendors)
synthesizer: endpoint/anthropic/claude-sonnet-4
chat_base_url: https://openrouter.ai/api/v1
chat_api_key_env: OPENROUTER_API_KEY
```

## Migration from removed providers

| Old spec                        | New spec                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------|
| `ollama/qwen2.5:14b`            | `synthesizer: endpoint/qwen2.5:14b` + `chat_base_url: http://localhost:11434/v1` |
| `cohere/command-a-03-2025`      | not directly supported — Cohere's chat endpoint isn't OpenAI-compatible. Use OpenRouter or another OpenAI-compatible gateway. |
| `anthropic/claude-sonnet-4`     | `synthesizer: endpoint/anthropic/claude-sonnet-4` via OpenRouter + `OPENROUTER_API_KEY` |

These migrations are surfaced in the `ValueError` raised when the
legacy spec is loaded at runtime.

## Programmatic usage

```python
from fitz_sage.llm.client import get_chat

chat = get_chat(
    "endpoint",
    config={"base_url": "http://localhost:8080/v1",
            "model": "qwen2.5-7b-instruct"},
)
response = chat.chat([{"role": "user", "content": "Hello"}])
```

## What `endpoint` is *not*

- It's not an embedding provider. fitz-sage has no embedding API in
  v0.12.0+.
- It's not a vector database. Retrieval indexes live in SQLite + FTS5
  (see [unified-storage.md](unified-storage.md)).
- It's not a tool-use / function-calling abstraction. Tool routing in
  KRAG happens above the chat layer.

## Why this matters

The honest-RAG thesis is that *lower recall plus correct abstention*
beats *higher recall plus occasional hallucination*. Embedding-based
retrieval had a known failure mode of surface-similar-but-wrong
candidates that the governance cascade then had to clean up. BM25 +
LLM rerank produces fewer such candidates. Cases that fall off the
recall curve are exactly the queries where `ABSTAIN` is the right
answer — which is what the pyrrho classifier does correctly. The
architecture is now lined up with the philosophy.

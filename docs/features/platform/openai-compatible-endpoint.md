<!-- docs/features/platform/openai-compatible-endpoint.md -->
# OpenAI-Compatible Endpoint Architecture

**Status:** the optional endpoint/cloud chat path. Managed local Qwen supplies
standard semantic query terms and optional background work; endpoint chat is
for optional synthesis, query intelligence, vision, and explicitly configured
chat-tier enhancements.

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

## Why a single protocol?

The endpoint architecture speaks the OpenAI HTTP protocol, which many modern
serving stacks expose:

- vLLM, LM Studio, TabbyAPI, Aphrodite, text-generation-webui
- Ollama (`/v1/` mode at `:11434/v1`)
- OpenAI, Together, Fireworks, Groq, DeepInfra, OpenRouter, Mistral La Plateforme

Optional chat and vision roles use this protocol. The separate `glm_ocr` parser
talks to Ollama's native API. Retrieval itself runs through SQLite FTS5, KRAG
routing, and the local ONNX cross-encoder reranker.

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
synthesizer: endpoint/qwen2.5-7b-instruct
chat_base_url: http://localhost:8080/v1
chat_api_key_env: null         # unauthenticated local server

rerank: onnx                   # INT8 ONNX cross-encoder, local on CPU
collection: default
```

Use role-specific provider fields (`query_intelligence`, `vision`, and
`synthesizer`) to mix local and cloud models. Managed Qwen work is internal and
does not need an endpoint.

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

## Programmatic usage

```python
from fitz_sage.llm.client import get_chat

chat = get_chat(
    "endpoint/qwen2.5-7b-instruct",
    config={"base_url": "http://localhost:8080/v1"},
)
response = chat.chat([{"role": "user", "content": "Hello"}])
```

## What `endpoint` is *not*

- It's not an embedding provider.
- It's not a vector database. Retrieval indexes live in SQLite + FTS5
  (see [unified-storage.md](unified-storage.md)).
- It's not a tool-use / function-calling abstraction. Tool routing in
  KRAG happens above the chat layer.

## Boundary

Endpoint providers never replace the default SQLite/BM25 retrieval path. Query
intelligence participates only when explicitly configured, and synthesis runs
only after governed evidence exists. Pyrrho remains the sole owner of the final
evidence verdict.

# OpenAI-Compatible Endpoint Architecture

**Status:** Canonical chat-protocol path since v0.12.0.

## TL;DR

There is **one** chat-protocol implementation in fitz-sage:
``OpenAICompatChat`` / ``OpenAICompatEmbedding`` / ``OpenAICompatVision``,
which speaks the OpenAI HTTP protocol against any compliant server.

Provider names are *configuration presets* on top of that single
implementation — not separate code paths:

| Spec                          | base_url                         | Auth                       | Use case                                            |
|-------------------------------|----------------------------------|----------------------------|-----------------------------------------------------|
| ``endpoint/<model>``          | required (user-supplied)         | NoAuth or ``api_key_env``  | local llama-server / vLLM / LM Studio / any cloud   |
| ``openai`` / ``openai/<m>``   | ``https://api.openai.com/v1``    | ``OPENAI_API_KEY``         | OpenAI public API                                   |
| ``azure_openai/<deployment>`` | required (tenant-specific)       | ``AZURE_OPENAI_API_KEY``   | Azure OpenAI                                        |
| ``enterprise/<m>``            | required                         | OAuth2 + downstream key    | Internal corporate gateway                          |

The legacy ``ollama``, ``cohere``, ``anthropic`` provider names have
been removed. Passing them raises an actionable migration error.

## Why

Historical fitz-sage shipped per-provider integrations (OpenAI, Cohere,
Anthropic, Ollama) as separate code paths. This forced two coupled
choices on every user:

1. Pick a chat provider that *also* has an embedding model.
2. Locally, swap models mid-query (Ollama unloads/reloads between chat
   and embedding) or run two SDK clients side-by-side.

Both are friction. The single-protocol architecture removes them by
speaking the OpenAI HTTP protocol — which virtually every modern serving
stack already exposes:

- ``llama-server`` (llama.cpp) — recommended local
- vLLM, LM Studio, TabbyAPI, Aphrodite, text-generation-webui
- Ollama (``/v1/`` mode)
- OpenAI, Together, Fireworks, Groq, DeepInfra, OpenRouter

## Authentication

The auth layer (``fitz_sage.llm.auth``) is the same for every preset:

- ``NoAuth`` — no headers (default for ``endpoint``).
- ``ApiKeyAuth`` — single header from an env var.
- ``M2MAuth`` — OAuth2 client credentials with auto-refresh.
- ``CompositeAuth`` (used by ``enterprise``) — M2M token + downstream
  API key.

For ``endpoint``, opt in to ApiKeyAuth via:

```yaml
chat_smart: endpoint/meta-llama-3.1-70b-instruct
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY
```

## Engine config (FitzKrag)

The KRAG engine config exposes per-role base URLs and api-key env vars
because chat, embedding, and vision often live behind different
servers (e.g. one llama-server per loaded model):

```yaml
fitz_krag:
  chat_fast: endpoint/qwen2.5-7b-instruct
  chat_balanced: endpoint/qwen2.5-7b-instruct
  chat_smart: endpoint/qwen2.5-7b-instruct
  embedding: endpoint/nomic-embed-text-v1.5

  chat_base_url: http://localhost:8080/v1       # llama-server (chat)
  embedding_base_url: http://localhost:8081/v1  # llama-server (embed)
  vision_base_url: null                         # falls back to chat
  chat_api_key_env: null
  embedding_api_key_env: null
  vision_api_key_env: null
```

These fields are wired to the LLM factories — the engine builds a
config dict only when the spec actually consumes ``base_url``
(``endpoint`` and ``enterprise``) so that switching to an OpenAI
preset doesn't accidentally route cloud calls to a local URL.

## Recommended local setup (llama.cpp)

```bash
# Chat server on port 8080
llama-server -m qwen2.5-7b-instruct-q4_k_m.gguf --port 8080 -c 8192

# Optional embedding server on port 8081
llama-server -m nomic-embed-text-v1.5.Q4_K_M.gguf --port 8081 --embeddings
```

Both stay hot in their own processes — no model thrashing, no SDK
swapping, no second key. fitz-sage's defaults match this layout out
of the box.

## Cloud quick reference

```yaml
# OpenAI (preset; no base_url needed)
chat_smart: openai/gpt-4o
embedding: openai/text-embedding-3-small
chat_base_url: null
embedding_base_url: null
# OPENAI_API_KEY in env

# Together (endpoint with API key)
chat_smart: endpoint/meta-llama-3.1-70b-instruct
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY

# OpenRouter (endpoint with API key)
chat_smart: endpoint/anthropic/claude-sonnet-4
chat_base_url: https://openrouter.ai/api/v1
chat_api_key_env: OPENROUTER_API_KEY
```

## Migration from removed providers

| Old spec                        | New spec                                                                               |
|---------------------------------|----------------------------------------------------------------------------------------|
| ``ollama/qwen2.5:14b``          | ``endpoint/qwen2.5-7b-instruct`` + ``base_url: http://localhost:11434/v1``             |
| ``cohere/command-a-03-2025``    | ``endpoint/command-a-03-2025`` + Cohere's ``/compatibility/v1`` URL + ``COHERE_API_KEY`` |
| ``anthropic/claude-sonnet-4``   | ``endpoint/anthropic/claude-sonnet-4`` via OpenRouter + ``OPENROUTER_API_KEY``         |

These migrations are also surfaced in the ``ValueError`` raised when
the legacy spec is passed at runtime.

## Programmatic usage

```python
from fitz_sage.llm import get_chat, get_embedder

chat = get_chat(
    "endpoint/qwen2.5-7b",
    config={"base_url": "http://localhost:8080/v1"},
)
response = chat.chat([{"role": "user", "content": "Hello"}])

embedder = get_embedder(
    "endpoint/nomic-embed-text",
    config={"base_url": "http://localhost:8081/v1"},
)
vector = embedder.embed("Hello world")
```

## What's not here yet

- **Chat-only retrieval mode.** Default retrieval still uses the
  embedding provider. The forthcoming change makes BM25 + KRAG
  routing + LLM-rerank the default, demoting embeddings to an
  optional accelerator. When that lands, the
  ``embedding_base_url`` / ``embedding_api_key_env`` fields become
  optional in practice.
- **CLI ``--endpoint`` flag.** The engine YAML is plumbed but
  ``fitz query --endpoint http://... --model ...`` does not yet
  exist.
- **README rewrite.** The top-level README still describes the older
  Ollama-or-cloud setup story.

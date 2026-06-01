<!-- docs/PLUGINS.md -->
# Plugin Development Guide

fitz-sage **v0.14.1+** has a much smaller plugin surface than earlier
versions: chat is mono-protocol (one OpenAI-compatible provider with
sugar presets), embedding/vector-db are gone, and the remaining plugin
types are Python modules wired by config.

This guide covers what's pluggable and how to add a new one.

---

## What's Pluggable

| Type            | Format | Location                                       | Selected by config              |
| --------------- | ------ | ---------------------------------------------- | -------------------------------- |
| Chat provider   | Python | `fitz_sage/llm/providers/`                     | `query_intelligence` / `synthesizer` / ... |
| Parser          | Python | `fitz_sage/ingestion/parser/plugins/`          | `parser:`                        |
| Chunker         | Python | `fitz_sage/ingestion/chunking/plugins/`        | `chunker:` / format auto-routing |
| Source          | Python | `fitz_sage/ingestion/source/plugins/`          | source spec at ingest time       |

There is **no embedding plugin, no vector-DB plugin, no rerank plugin**.
The reranker that does run (`OnnxReranker`) is an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) — see
`fitz_sage/llm/providers/onnx_reranker.py`.

---

## Feature Control

Features are controlled by **provider presence**, not boolean flags:

```yaml
# ENABLED (default) — INT8 ONNX cross-encoder
rerank: onnx
# or: rerank: onnx/<hf-model-id>  to swap in a different cross-encoder

# DISABLED — omit the key (or set null)
# rerank not set → no reranking step in the pipeline
```

The same pattern applies to vision (VLM): set `parser: docling_vision`
to bake the VLM into ingestion, or set `parser: cpu` / `parser: docling` /
`parser: glm_ocr` to skip it.

Governance is selected by the engine config's `governance:` field
(`pyrrho` or `null`) — it is not a plugin.

---

## Chat Provider Model

The LLM layer has exactly one canonical provider — **`endpoint`** —
that speaks OpenAI-compatible `/chat/completions`. The other names are
URL+auth presets over it:

| Spec form                              | Resolves to                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `endpoint` + `chat_base_url`           | the canonical form                                       |
| `openai/<model>`                       | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`            | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>`        | endpoint + M2M / mTLS / custom-CA auth                   |

Removed in v0.12.0 (raise `ValueError` with migration text):
`cohere`, `anthropic`, `ollama`. Point fitz-sage at the same model's
OpenAI-compatible endpoint instead — e.g. Ollama exposes one at
`http://localhost:11434/v1`.

### Built-in providers

| Provider     | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `endpoint`   | Canonical OpenAI-compatible chat (any URL)                 |
| `enterprise` | Same protocol + enterprise auth (M2M OAuth2, mTLS, CA bundle) |
| `onnx_reranker` | Internal — INT8 ONNX cross-encoder (gte-reranker-modernbert-base) |

### Role Providers

Role-specific provider fields are the primary control surface.
`chat_fast`, `chat_balanced`, and `chat_smart` tiers are optional slots for
advanced integrations that request a tiered chat factory directly:

```yaml
query_intelligence: endpoint/qwen2.5-3b-instruct
enricher: endpoint/qwen2.5-3b-instruct
summarizer: endpoint/qwen2.5-7b-instruct
synthesizer: endpoint/qwen2.5-32b-instruct
chat_base_url: http://localhost:8080/v1
chat_api_key_env: OPENAI_API_KEY    # omit for unauthenticated local servers
```

### Authentication

| Auth type     | Class            | Use case                                                |
| ------------- | ---------------- | ------------------------------------------------------- |
| None          | n/a              | Unauthenticated local server (llama.cpp, Ollama, LM Studio) |
| API key       | `ApiKeyAuth`     | OpenAI, Together, Groq, Anthropic-via-compat, ...       |
| M2M OAuth2    | `M2MAuth`        | Enterprise gateways using client-credentials flow       |
| Composite     | `CompositeAuth`  | Gateway with M2M bearer + downstream LLM API key        |

API-key env var names are arbitrary — set `chat_api_key_env` to
whichever env var holds yours.

See [`features/platform/enterprise-gateway.md`](features/platform/enterprise-gateway.md)
for the full M2M / mTLS story.

---

## Adding a Chat Provider

The expected case is that you don't need to — any OpenAI-compatible
server already works via `endpoint`. Add a new provider only when the
wire protocol isn't OpenAI-compatible (e.g. legacy enterprise gateways
with custom JSON shape). Then:

1. Create `fitz_sage/llm/providers/myprovider.py`.
2. Implement `ChatProvider`:

   ```python
   from typing import Any
   from fitz_sage.llm.types import ChatProvider

   class MyChat:
       def __init__(self, model: str, auth):
           self.model = model
           self.auth = auth

       def chat(self, messages: list[dict[str, Any]], **kwargs) -> str:
           ...
   ```
3. Register it in `fitz_sage/llm/config.py`'s provider factory
   (`_create_chat`).
4. Add a unit test that asserts the dispatch wire-up.

---

## Adding a Parser

Parsers go in `fitz_sage/ingestion/parser/plugins/<name>.py` and
inherit from `BaseParser`:

```python
from fitz_sage.ingestion.parser.base_parser import BaseParser
from fitz_sage.ingestion.parser.types import ParsedDocument

class MyParser(BaseParser):
    name = "my_parser"
    supported_extensions = (".myfmt",)

    def parse(self, path: str) -> ParsedDocument:
        ...
```

Auto-discovery walks `parser/plugins/` at import time, so dropping the
file is sufficient. Pick by setting `parser: my_parser` in config.

---

## Adding a Chunker

Chunkers live under `fitz_sage/ingestion/chunking/plugins/`. They
implement the `Chunker` protocol (`chunk(parsed: ParsedDocument) -> list[Chunk]`).
Routing is by file extension or content type — see
`ingestion/chunking/router.py`.

The default (markdown, plaintext, code, table) covers most cases.
Add a new plugin only when you have a format-specific structure to
preserve (e.g. a custom XML dialect).

---

## Enrichment Is Not a Plugin

Ingestion enrichment (keyword / entity / temporal extraction and the
L1/L2 hierarchy summaries) is built into the KRAG ingestion pipeline,
not a plugin surface. It is controlled by provider presence: `enricher:`
and `summarizer:`. See [ENRICHMENT.md](ENRICHMENT.md)
for the architecture and [`features/ingestion/hierarchical-rag.md`](features/ingestion/hierarchical-rag.md)
for the hierarchy details.

---

## Troubleshooting

### Unknown Provider

```
ValueError: Unknown chat provider: 'cohere'
```

That provider was removed in v0.12.0. Use `endpoint` with the
provider's OpenAI-compatible URL. Migration mapping for the most
common cases:

| Was              | Now                                                        |
| ---------------- | ---------------------------------------------------------- |
| `cohere`         | not available — pick an OpenAI-compatible model            |
| `ollama`         | `endpoint` with `chat_base_url: http://localhost:11434/v1` |
| `anthropic`      | not available — Claude isn't OpenAI-compatible             |

### Authentication Failed

- Set the env var named in `chat_api_key_env` (e.g. `OPENAI_API_KEY`).
- For enterprise auth, confirm all M2M fields are filled and the env
  vars referenced by `${...}` are set.

---

## See Also

- [ARCHITECTURE.md](ARCHITECTURE.md) — system overview
- [CONFIG.md](CONFIG.md) — full configuration reference
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — provider-presence pattern
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md) — the canonical chat provider
- [features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md) — M2M / mTLS / CA bundle

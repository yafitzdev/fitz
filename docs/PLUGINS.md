<!-- docs/PLUGINS.md -->
# Plugin Development Guide

fitz-sage plugins are Python modules wired by config. Required enrichment is
managed by the built-in ONNX provider, while optional endpoint/cloud chat uses
one OpenAI-compatible provider with URL and auth presets.

This guide covers what's pluggable and how to add a new one.

---

## What's Pluggable

| Type            | Format | Location                                       | Selected by config              |
| --------------- | ------ | ---------------------------------------------- | -------------------------------- |
| Chat provider   | Python | `fitz_sage/llm/providers/`                     | `query_intelligence` / `synthesizer` / ... |
| Parser          | Python | `fitz_sage/ingestion/parser/plugins/`          | `parser:`                        |
| Chunker         | Python | `fitz_sage/ingestion/chunking/plugins/`        | `chunker:` / format auto-routing |
| Source          | Python | `fitz_sage/ingestion/source/plugins/`          | source spec at ingest time       |

Reranking is built into the engine rather than exposed as a plugin. The
`OnnxReranker` is an INT8 ONNX cross-encoder
(`Alibaba-NLP/gte-reranker-modernbert-base` by default) — see
`fitz_sage/llm/providers/onnx_reranker.py`.

---

## Feature Control

Optional endpoint-backed features are controlled by **provider presence**, not
boolean flags. The retrieval backbone is different: managed Qwen enrichment,
ONNX reranking, and Pyrrho governance are the standard product path.

```yaml
# Standard retrieval path
rerank: onnx
governance: pyrrho
```

The provider-presence pattern applies to optional capabilities such as answer
synthesis, query intelligence, and vision parsing. For example, set
`synthesizer: endpoint/<model>` to enable generated answers, or leave
`synthesizer` unset to keep `fitz query` retrieval-only.

The same pattern applies to vision (VLM): set `parser: docling_vision`
to bake the VLM into ingestion, or set `parser: cpu` / `parser: docling` /
`parser: glm_ocr` to skip it.

Reranking and governance are not plugin toggles. Reranking is selected by the
engine config's `rerank:` field, governance is selected by `governance:`, and
both are part of the standard retrieval backbone.

---

## Chat Provider Model

The LLM layer has a managed Qwen3.5 0.8B ONNX runtime for required enrichment
and a canonical **`endpoint`** provider for optional OpenAI-compatible chat. The
other names are URL+auth presets over `endpoint`:

| Spec form                              | Resolves to                                              |
| -------------------------------------- | -------------------------------------------------------- |
| `onnx/qwen3.5-0.8b`                    | managed local Qwen3.5 0.8B ONNX runtime                  |
| `endpoint` + `chat_base_url`           | the canonical form                                       |
| `openai/<model>`                       | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`            | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>`        | endpoint + M2M / mTLS / custom-CA auth                   |

Provider specs must resolve to the OpenAI-compatible chat protocol. For local
servers such as Ollama, configure `endpoint` with the server's `/v1` URL, for
example `http://localhost:11434/v1`.

### Built-in providers

| Provider     | Purpose                                                    |
| ------------ | ---------------------------------------------------------- |
| `onnx`       | Managed local Qwen3.5 0.8B chat provider                   |
| `endpoint`   | Canonical OpenAI-compatible chat (any URL)                 |
| `enterprise` | Same protocol + enterprise auth (M2M OAuth2, mTLS, CA bundle) |
| `onnx_reranker` | Internal — INT8 ONNX cross-encoder (gte-reranker-modernbert-base) |

### Role Providers

Role-specific provider fields are the primary control surface for optional chat
paths. `chat_fast`, `chat_balanced`, and `chat_smart` tiers are optional slots
for advanced integrations that request a tiered chat factory directly:

```yaml
query_intelligence: endpoint/qwen2.5-3b-instruct
synthesizer: endpoint/qwen2.5-32b-instruct
chat_base_url: http://localhost:8080/v1
chat_api_key_env: OPENAI_API_KEY    # omit for unauthenticated local servers
```

### Authentication

| Auth type     | Class            | Use case                                                |
| ------------- | ---------------- | ------------------------------------------------------- |
| None          | n/a              | Managed ONNX or unauthenticated local endpoint server       |
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
wire protocol isn't OpenAI-compatible, such as a custom enterprise gateway
with a nonstandard JSON shape. Then:

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
not a plugin surface and not user-configurable. It always uses the managed
Qwen3.5 0.8B ONNX runtime. See [ENRICHMENT.md](ENRICHMENT.md)
for the architecture and [`features/ingestion/hierarchical-rag.md`](features/ingestion/hierarchical-rag.md)
for the hierarchy details.

---

## Troubleshooting

### Unknown Provider

```
ValueError: Unknown chat provider: '<provider>'
```

Use `endpoint` with the configured server's OpenAI-compatible URL. For
example, Ollama exposes `http://localhost:11434/v1`.

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

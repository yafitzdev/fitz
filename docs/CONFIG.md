<!-- docs/CONFIG.md -->
# Configuration Reference

fitz-sage **v0.14.1+**. Engine config lives at
`~/.fitz/config/<engine>.yaml` (auto-created on first run).
Storage lives at `~/.fitz/sqlite/fitz_<collection>.db` (auto-created
on first ingest).

---

## Minimal config

```yaml
# ~/.fitz/config/fitz_krag.yaml
collection: default
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
enricher: null
summarizer: null
```

This is enough for `fitz retrieve` and `fitz_sage.evidence(...)`: no API key and
no local chat server are required.

To enable synthesized answers through a hosted endpoint:

```yaml
synthesizer: endpoint/gpt-4o
chat_base_url: https://api.openai.com/v1
chat_api_key_env: OPENAI_API_KEY
collection: default
```

Or pass the provider per invocation:

```bash
fitz answer "What is X?" \
  --endpoint https://api.openai.com/v1 \
  --model gpt-4o \
  --api-key-env OPENAI_API_KEY
```

---

## Optional LLM Providers

The retrieval path does not require chat. Chat tiers are provider definitions
used only when another config key opts into an LLM-backed stage.

| Key             | Typical use when referenced by a feature provider |
| --------------- | -------------------------------------------------- |
| `chat_fast`     | Query-intelligence enhancement, enrichment         |
| `chat_balanced` | Summaries, table query helpers                     |
| `chat_smart`    | Synthesis-oriented defaults                        |

Each tier takes a provider/model spec. For `endpoint`, the model name is the
part after the slash and `chat_base_url` supplies the OpenAI-compatible URL:

```yaml
chat_smart: endpoint/qwen2.5-32b
chat_base_url: http://localhost:8080/v1
```

If `chat_base_url` is shared across tiers (the common case), set it
once at the top level.

---

## Chat provider model

The canonical provider is **`endpoint`** — OpenAI-compatible HTTP.
Everything else is a preset:

| Spec form                       | Resolves to                                              |
| ------------------------------- | -------------------------------------------------------- |
| `endpoint` + `chat_base_url`    | the canonical form                                       |
| `openai/<model>`                | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`     | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>` | endpoint + M2M / mTLS / custom-CA auth                   |

Removed names (raise `ValueError` with migration text): `cohere`,
`anthropic`, `ollama`. Point fitz-sage at the OpenAI-compatible URL
for those backends instead (Ollama exposes one at
`http://localhost:11434/v1`).

---

## Feature control

Features are switched on by **provider presence**, not boolean flags:

| Feature            | Enabled when                             | Disabled when                       |
| ------------------ | ---------------------------------------- | ----------------------------------- |
| ONNX reranker      | `rerank: onnx` (default)                 | `rerank: null`                      |
| Governance         | `governance: pyrrho` (default)           | `governance: null`                  |
| Query intelligence | `query_intelligence: <provider/model>`   | `query_intelligence: null`          |
| Answer synthesis   | `synthesizer: <provider/model>`          | `synthesizer: null`                 |
| Enrichment         | `enricher: <provider/model>`             | `enricher: null`                    |
| Hierarchy summaries | `summarizer: <provider/model>`          | `summarizer: null`                  |
| VLM in parser      | `parser: docling_vision` + `vision:` set | `parser: cpu`, `parser: docling`, or `parser: glm_ocr` |

---

## Storage

```yaml
collection: default          # active collection
```

That's the entire storage surface. SQLite + FTS5 + WAL, one `.db` per
collection. Storage paths are derived automatically — one
`fitz_<collection>.db` under the workspace `sqlite/` directory; there
is no settable storage-path config key. No `vector_db`, no
`connection_string`, no `pgvector_kwargs`.
See [features/platform/unified-storage.md](features/platform/unified-storage.md)
for the schema and pragmas.

---

## Parser

```yaml
parser: cpu                # CPU pypdfium2 text-layer parser (default)
# parser: docling          # Docling structure extraction
# parser: docling_vision   # Docling + VLM for figure descriptions
# parser: glm_ocr          # hybrid pypdfium2 + GLM-OCR, handles scanned pages
```

| Parser           | Speed (100pg PDF) | Scanned pages | Install                            |
| ---------------- | ----------------- | ------------- | ---------------------------------- |
| `cpu`            | seconds to minutes | no            | base install                       |
| `docling`        | ~21 min           | no            | `pip install fitz-sage[docs]`      |
| `docling_vision` | ~21 min + VLM     | VLM figures   | `[docs]` + `vision:` set           |
| `glm_ocr`        | ~28 s             | yes (GLM-OCR) | base install                       |

Only `cpu`, `docling`, `docling_vision`, and `glm_ocr` are selectable
`parser:` values. `lightweight` is an automatic ImportError fallback (not
selectable), and plain-text / CSV files are routed by extension to
built-in plugins — they are not `parser:` options.

---

## Authentication

API keys are needed only for optional hosted providers. They are read from
environment variables — never put them in the config file. Name the env var with
`chat_api_key_env`:

```yaml
chat_api_key_env: OPENAI_API_KEY
```

| Service          | Common env var       |
| ---------------- | -------------------- |
| OpenAI           | `OPENAI_API_KEY`     |
| Together         | `TOGETHER_API_KEY`   |
| Groq             | `GROQ_API_KEY`       |
| Mistral La Plateforme | `MISTRAL_API_KEY` |
| Local llama.cpp / LM Studio / Ollama | (no key) |

For enterprise (M2M / mTLS) deployments see
[features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md).

---

## Retrieval knobs

```yaml
top_addresses: 50      # how many candidates to fetch from FTS5 (default 50)
top_read: 50           # how many to read into context after rerank (default 50)
retrieval_workers: 4   # max retrieval strategies run concurrently; set to 1 to serialize LLM calls for single-model local servers (LM Studio, llama-server)
governance: pyrrho
strict_grounding: false
```

KRAG also exposes detection, multi-hop, and rewriter switches under
their respective config blocks. The defaults are the smoke-tested
baseline.

---

## Per-invocation overrides

The CLI accepts overrides without editing the config file:

```bash
fitz answer "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --model meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY
```

`fitz answer` uses these flags to configure the synthesizer for that invocation.
`fitz retrieve` ignores chat endpoint flags because it does not synthesize.

---

## See Also

- [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) — full example configs by deployment
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — the provider-presence pattern
- [CLI.md](CLI.md) — CLI reference
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md)
- [features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md)

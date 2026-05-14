# Configuration Reference

fitz-sage **v0.12.0+**. Engine config lives at
`~/.fitz/config/<engine>.yaml` (auto-created on first `fitz init`).
Storage lives at `~/.fitz/sqlite/fitz_<collection>.db` (auto-created
on first ingest).

---

## Minimal config

```yaml
# ~/.fitz/config/fitz_krag.yaml — local llama.cpp / Ollama / LM Studio
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:8080/v1
chat_smart_model: qwen2.5-7b-instruct
collection: default
```

To talk to a hosted endpoint:

```yaml
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: https://api.openai.com/v1
chat_api_key_env: OPENAI_API_KEY
chat_smart_model: gpt-4o
chat_balanced_model: gpt-4o-mini
chat_fast_model: gpt-4o-mini
collection: default
```

Or override the entire chat stack per-invocation with CLI flags
(`--endpoint`, `--model`, `--api-key-env`) — no config file needed.

---

## LLM tiers

| Key             | Purpose                          | Typical use                                   |
| --------------- | -------------------------------- | --------------------------------------------- |
| `chat_fast`     | Cheap/fast                       | Classification, detection, query rewriting    |
| `chat_balanced` | Middle tier                      | SQL generation, table queries, enrichment     |
| `chat_smart`    | Best reasoning                   | Answer synthesis                              |

Each tier takes a provider spec (almost always `endpoint`). Pair it
with a model name via the per-tier `*_model` key, or with a
provider-prefixed model in the spec itself:

```yaml
chat_smart: endpoint            # provider only
chat_smart_model: qwen2.5-32b   # model name passed to the endpoint
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

| Feature       | Enabled when                                | Disabled when                       |
| ------------- | ------------------------------------------- | ----------------------------------- |
| ONNX reranker | `rerank: onnx` (default)                    | `rerank: null` (or omitted)         |
| VLM in parser | `parser: docling_vision` + `vision:` set    | `parser: docling` or `parser: glm_ocr` |
| Enrichment    | `chat_*` configured (always-on otherwise)   | no chat provider                    |

The one boolean exception is `enable_guardrails`, used by the smoke
test to bypass constraints for raw retrieval timing.

---

## Storage

```yaml
collection: default          # active collection
# storage_path: ~/.fitz/sqlite  # override (defaults to <workspace>/sqlite)
```

That's the entire storage surface. SQLite + FTS5 + WAL, one `.db` per
collection. No `vector_db`, no `connection_string`, no `pgvector_kwargs`.
See [features/platform/unified-storage.md](features/platform/unified-storage.md)
for the schema and pragmas.

---

## Parser

```yaml
parser: glm_ocr            # hybrid pdfplumber + GLM-OCR (fast, default)
# parser: docling          # Docling — slower, no OCR
# parser: docling_vision   # Docling + VLM for figure descriptions
```

| Parser           | Speed (100pg PDF) | Scanned pages | Install                            |
| ---------------- | ----------------- | ------------- | ---------------------------------- |
| `glm_ocr`        | ~28 s             | yes (GLM-OCR) | base install                       |
| `docling`        | ~21 min           | no            | `pip install fitz-sage[docs]`      |
| `docling_vision` | ~21 min + VLM     | VLM figures   | `[docs]` + `vision:` set           |
| `lightweight`    | ms                | no            | base (uses `pypdf` only)           |
| `plaintext`      | ms                | n/a           | base                               |
| `csv`            | ms                | n/a           | base                               |

---

## Authentication

API keys are read from environment variables — never put them in the
config file. Name the env var with `chat_api_key_env`:

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
top_addresses: 20      # how many candidates to fetch from FTS5 (default 20)
top_read: 10           # how many to read into context after rerank (default 10)
enable_guardrails: true
strict_grounding: false
```

KRAG also exposes detection, multi-hop, and rewriter switches under
their respective config blocks. The defaults match the smoke baseline
that ships in `.smoke_test/`.

---

## Per-invocation overrides

The CLI accepts overrides without editing the config file:

```bash
fitz query "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --model meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY
```

`--endpoint` overrides `chat_base_url`, `--model` overrides the smart
tier's model, `--api-key-env` overrides `chat_api_key_env`.

---

## See Also

- [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) — full example configs by deployment
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — the provider-presence pattern
- [CLI.md](CLI.md) — CLI reference
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md)
- [features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md)

<!-- docs/CONFIG.md -->
# Configuration Reference

Engine config lives at
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
chat_base_url: http://127.0.0.1:8080/v1
```

This is enough for `fitz query`, `fitz retrieve`, and `fitz_sage.evidence(...)` with local
enrichment: no hosted API key and no external inference server are required.
On first ingestion, fitz-sage downloads the managed Qwen3.5 0.8B ONNX weights
into the Hugging Face cache and runs them on CPU.

To enable synthesized answers through a hosted endpoint:

```yaml
synthesizer: endpoint/gpt-4o
chat_base_url: https://api.openai.com/v1
chat_api_key_env: OPENAI_API_KEY
collection: default
```

Or pass the provider per invocation:

```bash
fitz answer "What is X?" --synthesizer openai/gpt-4o
```

---

## LLM Providers

Role-specific provider fields bind LLM-backed stages:

| Key                  | Typical use                                      |
| -------------------- | ------------------------------------------------ |
| `query_intelligence` | Optional query-prep enhancement                  |
| `synthesizer`        | Optional answer generation                       |

Required keyword/entity enrichment, hierarchy summaries, and default semantic
query keywords always use Fitz's managed Qwen3.5 0.8B ONNX runtime. There is no
config key for that internal model. Optional LLM-backed roles take a provider/model spec. For `endpoint`,
the model name is the part after the slash and `chat_base_url` supplies the
OpenAI-compatible URL:

```yaml
synthesizer: endpoint/qwen2.5-32b
chat_base_url: http://localhost:8080/v1
```

If `chat_base_url` is shared across roles (the common case), set it once at
the top level. `chat_fast`, `chat_balanced`, and `chat_smart` are optional
low-level tier slots for advanced integrations that request a tiered chat
factory directly.

---

## Chat provider model

Required enrichment uses Qwen3.5 0.8B through an in-process CPU ONNX runtime
managed by fitz-sage. The model is downloaded on first ingest if missing.
Optional synthesis, query intelligence, and vision can use **`endpoint`** or
the cloud/enterprise presets:

| Spec form                       | Resolves to                                              |
| ------------------------------- | -------------------------------------------------------- |
| `onnx/qwen3.5-0.8b`             | managed local Qwen3.5 0.8B ONNX runtime                  |
| `endpoint` + `chat_base_url`    | the canonical form                                       |
| `openai/<model>`                | endpoint pointing at `https://api.openai.com/v1`         |
| `azure_openai/<deployment>`     | endpoint with Azure deployment URL                       |
| `enterprise/<provider>/<model>` | endpoint + M2M / mTLS / custom-CA auth                   |

Provider specs must resolve to the OpenAI-compatible chat protocol.
For local servers such as Ollama, use `endpoint` with the server's
`/v1` URL, for example `http://localhost:11434/v1`.

---

## Feature control

The retrieval backbone is not optional: managed Qwen enrichment, the ONNX
reranker, and Pyrrho governance always run in the standard product path.
Optional endpoint-backed features are switched on by **provider presence**,
not boolean flags:

| Feature            | Standard / enabled when                  | Disabled when                       |
| ------------------ | ---------------------------------------- | ----------------------------------- |
| Managed enrichment | always uses local `onnx/qwen3.5-0.8b`    | not disabled                        |
| ONNX reranker      | `rerank: onnx` (default)                 | not disabled                        |
| Governance         | `governance: pyrrho` (default)           | not disabled                        |
| Query intelligence | `query_intelligence: <provider/model>`   | `query_intelligence: null`          |
| Answer synthesis   | `synthesizer: <provider/model>`          | `synthesizer: null`                 |
| VLM in parser      | `parser: docling_vision` + `vision:` set | `parser: cpu`, `parser: docling`, or `parser: glm_ocr` |

---

## Storage

```yaml
collection: default          # active collection
```

That's the entire storage surface. SQLite + FTS5 + WAL, one `.db` per
collection. Storage paths are derived automatically — one
`fitz_<collection>.db` under the workspace `sqlite/` directory; there
is no settable storage-path config key.
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

API keys are needed only for hosted providers. They are read from
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
| Managed ONNX enrichment | (no key) |
| Local endpoint server / LM Studio / Ollama | (no key) |

For enterprise (M2M / mTLS) deployments see
[features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md).

---

## Retrieval knobs

```yaml
top_addresses: 50      # candidates fetched during broad recall
top_read: 50           # candidates read after rerank
retrieval_workers: 4   # max retrieval strategies run concurrently
governance: pyrrho
rerank: onnx
```

Query intelligence defaults to deterministic detection plus managed Qwen
semantic keywords. Set `query_intelligence:` only when you want an optional
endpoint-backed rewrite/analyze/detect bus.

---

## Synthesis Knobs

```yaml
synthesizer: openai/gpt-4o
max_answer_tokens: 512       # general answer cap
short_answer_tokens: 192     # factual-question cap
strict_grounding: true
```

The optional synthesizer prompt is concise by default. For narrow factual
questions, fitz-sage uses the smaller `short_answer_tokens` cap so small local
models do not spend most of the query time writing long prose.

---

## Per-invocation overrides

The CLI accepts overrides without editing the config file:

```bash
fitz answer "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --synthesizer endpoint/meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY
```

`fitz answer` uses these flags to configure the synthesizer for that invocation.
`fitz query` and `fitz retrieve` ignore synthesis override flags because they do
not synthesize.

---

## See Also

- [CONFIG_EXAMPLES.md](CONFIG_EXAMPLES.md) — full example configs by deployment
- [RETRIEVAL_PIPELINE.md](RETRIEVAL_PIPELINE.md) — retrieval and enrichment flow
- [FEATURE_CONTROL.md](FEATURE_CONTROL.md) — the provider-presence pattern
- [CLI.md](CLI.md) — CLI reference
- [features/platform/openai-compatible-endpoint.md](features/platform/openai-compatible-endpoint.md)
- [features/platform/enterprise-gateway.md](features/platform/enterprise-gateway.md)

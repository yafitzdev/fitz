<!-- docs/CONFIG_EXAMPLES.md -->
# Configuration Examples

Working configs for the v0.12.0+ single-protocol / SQLite world. The
schema rules:

- **String specs** instead of nested dicts (`synthesizer: endpoint/gpt-4o`,
  not a provider block).
- **Provider presence** controls features (`synthesizer: null` disables
  answer generation; `rerank: null` disables the reranker).
- **Sensible defaults** — `collection` is the only thing every config
  must set; the rest can be overridden per-invocation via CLI flags.

---

## Minimal: retrieval-only

```yaml
# ~/.fitz/config/fitz_krag.yaml
collection: my_docs
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
enricher: null
summarizer: null
```

No API key or chat server is needed. Storage is SQLite + FTS5, auto-managed
under `~/.fitz/sqlite/`.

---

## Optional synthesis: local llama.cpp / LM Studio

```yaml
collection: my_docs
synthesizer: endpoint/qwen3.5-0.8b
chat_base_url: http://localhost:8080/v1
max_answer_tokens: 512
short_answer_tokens: 192
```

No API key needed — local servers are usually unauthenticated by default.
The shorter factual-question cap keeps small local models from writing long
answers for simple lookup questions.

---

## Optional synthesis: Ollama (OpenAI-compatible mode)

Ollama exposes the OpenAI protocol at `:11434/v1`:

```yaml
synthesizer: endpoint/qwen2.5:7b-instruct
chat_base_url: http://localhost:11434/v1
collection: my_docs
```

---

## Optional synthesis: OpenAI cloud

```yaml
synthesizer: endpoint/gpt-4o
chat_base_url: https://api.openai.com/v1
chat_api_key_env: OPENAI_API_KEY
collection: my_docs
```

Set the env var: `export OPENAI_API_KEY=...`.

---

## Together / Groq / Mistral La Plateforme

Any OpenAI-compatible cloud works with the same shape:

```yaml
# Together
synthesizer: endpoint/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY

# Groq
synthesizer: endpoint/llama-3.1-70b-versatile
chat_base_url: https://api.groq.com/openai/v1
chat_api_key_env: GROQ_API_KEY

# Mistral
synthesizer: endpoint/mistral-large-latest
chat_base_url: https://api.mistral.ai/v1
chat_api_key_env: MISTRAL_API_KEY
```

---

## Mixed local + cloud (cost-optimized)

Cheap local model for optional enrichment and query intelligence, smart cloud
model for optional synthesis:

```yaml
collection: my_docs
chat_base_url: http://localhost:8080/v1
chat_api_key_env: OPENAI_API_KEY

query_intelligence: endpoint/qwen2.5-3b-instruct
enricher: endpoint/qwen2.5-3b-instruct
summarizer: endpoint/qwen2.5-7b-instruct
synthesizer: openai/gpt-4o
```

---

## With ONNX cross-encoder reranker

```yaml
# Default — INT8 ONNX cross-encoder (gte-reranker-modernbert-base, 149M)
rerank: onnx
# Or pick a different cross-encoder:
# rerank: onnx/BAAI/bge-reranker-base
# rerank: onnx/jinaai/jina-reranker-v3
collection: my_docs
```

The reranker runs locally on CPU (~30–100 ms for 10–20 candidates) and
does not consume the chat endpoint.

---

## With VLM in the parser

```yaml
vision: endpoint/gpt-4o           # any OpenAI-compatible vision model
vision_base_url: https://api.openai.com/v1
vision_api_key_env: OPENAI_API_KEY

parser: docling_vision            # the parser that consults `vision:`
collection: my_docs
```

Use `parser: cpu` (the default), `parser: docling`, or `parser: glm_ocr`
to skip the VLM and avoid the cost.

---

## Optional Qwen 0.8B background enrichment

```yaml
collection: my_docs
chat_base_url: http://localhost:1234/v1
enricher: endpoint/qwen3.5-0.8b
summarizer: endpoint/qwen3.5-0.8b
summary_batch_size: 15
```

Use a Q4_K_M quantized small model when CPU-only enrichment is acceptable.
Retrieval continues to work if these providers are omitted or offline.

---

## Production: enterprise gateway with M2M + custom CA

```yaml
collection: production_docs
synthesizer: enterprise/openai/gpt-4o
query_intelligence: enterprise/openai/gpt-4o-mini
enricher: enterprise/openai/gpt-4o-mini
summarizer: enterprise/openai/gpt-4o-mini

auth:
  type: enterprise
  base_url: https://llm.corp.internal/v1
  token_url: https://auth.corp.internal/oauth/token
  client_id: ${CORP_CLIENT_ID}
  client_secret: ${CORP_CLIENT_SECRET}
  scope: llm.access
  llm_api_key_env: CORP_LLM_API_KEY
  llm_api_key_header: X-Api-Key
  cert_path: /etc/ssl/corp-ca-bundle.crt
  client_cert_path: /etc/ssl/client.crt
  client_key_path: /etc/ssl/client.key

# Storage stays local SQLite — one .db per collection
# under ~/.fitz/sqlite/. No DB knobs to configure.
```

Set env vars: `CORP_CLIENT_ID`, `CORP_CLIENT_SECRET`,
`CORP_LLM_API_KEY`.

---

## Per-invocation overrides (no config edit)

```bash
# Configure synthesis just for this answer
fitz answer "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --synthesizer endpoint/meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY \
  --source ./docs
```

`--synthesizer`, `--endpoint`, and `--api-key-env` configure the synthesizer
for that invocation.

---

## Programmatic, zero-config

```python
from fitz_sage.engines.fitz_krag import FitzKragEngine, FitzKragConfig
from fitz_sage.core import Query

cfg = FitzKragConfig(
    collection="my_docs",
    synthesizer=None,
    query_intelligence=None,
    enricher=None,
    summarizer=None,
)
engine = FitzKragEngine(cfg)
pack = engine.evidence(Query(text="What is quantum computing?"))
print(pack.mode, [item.file_path for item in pack.items])
```

Only `collection` is strictly required; everything else can come from
defaults or CLI flags.

---

## What's gone in v0.12.0

If you're migrating from v0.11.x:

| Removed key                    | Replacement                                              |
| ------------------------------ | -------------------------------------------------------- |
| `embedding: ...`               | (deleted — no embeddings anymore)                        |
| `vector_db: pgvector`          | (deleted — SQLite + FTS5 is the only storage)            |
| `vector_db_kwargs: ...`        | (deleted — no DB knobs)                                  |
| `cloud: {enabled: true, ...}`  | (deleted — Fitz Cloud cache removed)                     |
| `chat_smart: cohere/...`       | `synthesizer: endpoint/...`, point `chat_base_url` at the API |
| `chat_smart: ollama/...`       | `synthesizer: endpoint/...`, `chat_base_url: http://localhost:11434/v1` |
| `chat_smart: anthropic/...`    | not directly available — pick an OpenAI-compatible model |

Loading a config with any of the deleted keys raises a `ValueError`
with the migration message.

---

## What's gone in v0.13.1

| Removed key                  | Replacement          |
| ---------------------------- | -------------------- |
| `enable_guardrails: true`    | `governance: pyrrho` |
| `enable_guardrails: false`   | `governance: null`   |

`enable_guardrails` is replaced by `governance`, which follows the
provider-presence pattern (`rerank`, `vision`, `parser`). Loading a
config with the old key raises a `ValueError` with the migration message.

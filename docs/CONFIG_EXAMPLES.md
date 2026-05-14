# Configuration Examples

Working configs for the v0.12.0+ single-protocol / SQLite world. The
schema rules:

- **String specs** instead of nested dicts (`chat_smart: endpoint`, not
  a provider block).
- **Provider presence** controls features (omit `rerank:` to disable
  the reranker; there's no `enabled` flag).
- **Sensible defaults** — `collection` is the only thing every config
  must set; the rest can be overridden per-invocation via CLI flags.

---

## Minimal: local llama.cpp / LM Studio

```yaml
# ~/.fitz/config/fitz_krag.yaml
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:8080/v1
chat_smart_model: qwen2.5-7b-instruct
collection: my_docs
```

No API key needed — the local server is unauthenticated by default.
Storage is SQLite + FTS5, auto-managed under `~/.fitz/sqlite/`.

---

## Minimal: Ollama (OpenAI-compatible mode)

Ollama exposes the OpenAI protocol at `:11434/v1`:

```yaml
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: http://localhost:11434/v1
chat_smart_model: qwen2.5:7b-instruct
collection: my_docs
```

---

## Minimal: OpenAI cloud

```yaml
chat_fast: endpoint
chat_balanced: endpoint
chat_smart: endpoint
chat_base_url: https://api.openai.com/v1
chat_api_key_env: OPENAI_API_KEY
chat_smart_model: gpt-4o
chat_balanced_model: gpt-4o-mini
chat_fast_model: gpt-4o-mini
collection: my_docs
```

Set the env var: `export OPENAI_API_KEY=...`.

---

## Together / Groq / Mistral La Plateforme

Any OpenAI-compatible cloud works with the same shape:

```yaml
# Together
chat_smart: endpoint
chat_base_url: https://api.together.xyz/v1
chat_api_key_env: TOGETHER_API_KEY
chat_smart_model: meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo

# Groq
chat_smart: endpoint
chat_base_url: https://api.groq.com/openai/v1
chat_api_key_env: GROQ_API_KEY
chat_smart_model: llama-3.1-70b-versatile

# Mistral
chat_smart: endpoint
chat_base_url: https://api.mistral.ai/v1
chat_api_key_env: MISTRAL_API_KEY
chat_smart_model: mistral-large-latest
```

---

## Mixed local + cloud (cost-optimised)

Cheap local model for the bulk of the work, smart cloud model for
synthesis:

```yaml
chat_fast: endpoint
chat_fast_base_url: http://localhost:8080/v1
chat_fast_model: qwen2.5-3b-instruct

chat_balanced: endpoint
chat_balanced_base_url: http://localhost:8080/v1
chat_balanced_model: qwen2.5-7b-instruct

chat_smart: endpoint
chat_smart_base_url: https://api.openai.com/v1
chat_smart_api_key_env: OPENAI_API_KEY
chat_smart_model: gpt-4o

collection: my_docs
```

(Per-tier `*_base_url`, `*_model`, `*_api_key_env` keys override the
shared top-level ones for that tier only.)

---

## With LLM reranker

```yaml
chat_smart: endpoint
chat_base_url: http://localhost:8080/v1
chat_smart_model: qwen2.5-7b-instruct

rerank: endpoint/llmreranker      # presence enables the reranker step
collection: my_docs
```

The reranker uses the same chat endpoint — it asks the model to score
documents in a single JSON-returning call.

---

## With VLM in the parser

```yaml
chat_smart: endpoint
chat_base_url: http://localhost:8080/v1
chat_smart_model: qwen2.5-7b-instruct

vision: endpoint                  # any OpenAI-compatible vision model
vision_base_url: https://api.openai.com/v1
vision_api_key_env: OPENAI_API_KEY
vision_model: gpt-4o

parser: docling_vision            # the parser that consults `vision:`
collection: my_docs
```

Use `parser: docling` or `parser: glm_ocr` to skip the VLM and avoid the cost.

---

## Production: enterprise gateway with M2M + custom CA

```yaml
chat_smart: enterprise/openai/gpt-4o
chat_balanced: enterprise/openai/gpt-4o-mini
chat_fast: enterprise/openai/gpt-4o-mini
collection: production_docs

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
# Override the smart endpoint just for this query
fitz query "What is X?" \
  --endpoint https://api.together.xyz/v1 \
  --model meta-llama-3.1-70b \
  --api-key-env TOGETHER_API_KEY \
  --source ./docs
```

`--endpoint` overrides `chat_base_url`, `--model` overrides the smart
tier's model, `--api-key-env` overrides `chat_api_key_env`.

---

## Programmatic, zero-config

```python
from fitz_sage.engines.fitz_krag import FitzKragEngine, FitzKragConfig
from fitz_sage.core import Query

cfg = FitzKragConfig(
    chat_fast="endpoint",
    chat_balanced="endpoint",
    chat_smart="endpoint",
    chat_base_url="http://localhost:8080/v1",
    chat_smart_model="qwen2.5-7b-instruct",
    collection="my_docs",
)
engine = FitzKragEngine(cfg)
answer = engine.answer(Query(text="What is quantum computing?"))
print(answer.text, answer.mode, [p.address for p in answer.provenance])
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
| `chat_smart: cohere/...`       | `chat_smart: endpoint`, point `chat_base_url` at the API |
| `chat_smart: ollama/...`       | `chat_smart: endpoint`, `chat_base_url: http://localhost:11434/v1` |
| `chat_smart: anthropic/...`    | not directly available — pick an OpenAI-compatible model |

Loading a config with any of the deleted keys raises a `ValueError`
with the migration message.

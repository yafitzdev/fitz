<!-- docs/CONFIG_EXAMPLES.md -->
# Configuration Examples

Working configs for the managed-ONNX / SQLite world. The schema rules:

- **String specs** instead of nested dicts (`synthesizer: endpoint/gpt-4o`,
  not a provider block).
- **Provider presence** controls optional endpoint-backed features
  (`synthesizer: null` means no generated answer; `query_intelligence: null`
  means deterministic query prep plus managed Qwen semantic keywords).
- **Retrieval defaults are real defaults** — managed Qwen enrichment, Pyrrho
  governance, and the ONNX reranker are the product path.
- **Sensible defaults** — `collection` is the only thing every config
  must set; the rest can be overridden per-invocation via CLI flags.

---

## Minimal: retrieval-first local setup

```yaml
# ~/.fitz/config/fitz_krag.yaml
collection: my_docs
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
chat_base_url: http://127.0.0.1:8080/v1
```

No hosted API key or external inference server is needed for `fitz query`,
`fitz retrieve`, or `fitz_sage.evidence(...)`. On first ingestion,
fitz-sage downloads the managed Qwen3 0.6B ONNX GenAI weights into the Hugging Face
cache and runs them locally on CPU.

---

## Optional synthesis: local OpenAI-compatible endpoint

```yaml
collection: my_docs
synthesizer: endpoint/qwen2.5-7b-instruct
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

Required internal Qwen enrichment, optional local query intelligence, and a
smart cloud model for optional synthesis:

```yaml
collection: my_docs
chat_base_url: http://localhost:8080/v1

query_intelligence: endpoint/qwen2.5-7b-instruct
synthesizer: openai/gpt-4o
```

---

## Standard ONNX cross-encoder reranker

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

## Standard Qwen 0.6B enrichment

```yaml
collection: my_docs
summary_batch_size: 15
```

Qwen3 0.6B ONNX GenAI is the standard local enrichment model. Fitz downloads it
on first ingest if missing and runs it on CPU through `onnxruntime-genai`.
Runtime failures are surfaced instead of silently weakening the retrieval index.

---

## Production: enterprise gateway with M2M + custom CA

```yaml
collection: production_docs
synthesizer: enterprise/openai/gpt-4o
query_intelligence: enterprise/openai/gpt-4o-mini

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
)
engine = FitzKragEngine(cfg)
pack = engine.evidence(Query(text="What is quantum computing?"))
print(pack.mode, [item.file_path for item in pack.items])
```

Only `collection` is strictly required by the schema. Enrichment and
summarization use the managed Qwen3 0.6B ONNX GenAI runtime automatically.

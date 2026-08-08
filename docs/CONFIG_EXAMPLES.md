<!-- docs/CONFIG_EXAMPLES.md -->
# Configuration Examples

Working configs for the managed-ONNX / SQLite world. The schema rules:

- **String specs** instead of nested dicts (`synthesizer: endpoint/gpt-4o`,
  not a provider block).
- **Provider presence** controls optional endpoint-backed features
  (`synthesizer: null` means no generated answer; `query_intelligence: null`
  means deterministic query prep plus managed Qwen semantic keywords).
- **Governance is mandatory** — bare `pyrrho` uses the accepted immutable
  default; advanced users can select a local or commit-pinned package.
- **Sensible defaults** — `collection` is the only required field. Selected
  synthesis provider fields can be overridden by `fitz answer`; other settings
  use schema defaults or the YAML file.

---

## Minimal: retrieval-first local setup

```yaml
# .fitz/config.yaml
collection: my_docs
parser: cpu
rerank: onnx
governance: pyrrho
query_intelligence: null
synthesizer: null
chat_base_url: http://127.0.0.1:8080/v1
```

No hosted API key or external inference server is needed for `fitz retrieve`
or `fitz_sage.evidence(...)`. See
[Managed Models](MANAGED_MODELS.md) for exact local model IDs, runtimes, cache
locations, and the smoke command.

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

## Mixed local + cloud

Managed local Qwen query expansion, optional endpoint query intelligence, and
a cloud model for optional synthesis:

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
rerank_candidates: 32
# A custom repository must provide a compatible tokenizer and
# onnx/model_int8.onnx artifact:
# rerank: onnx/owner/compatible-reranker
collection: my_docs
```

The reranker runs locally on CPU over a bounded candidate prefix and does not
consume the chat endpoint.

---

## With VLM in the parser

```yaml
vision: endpoint/gpt-4o           # any OpenAI-compatible vision model
vision_base_url: https://api.openai.com/v1
vision_api_key_env: OPENAI_API_KEY

parser: docling_vision            # the parser that consults `vision:`
collection: my_docs
```

Use `parser: cpu` (the default) or `parser: docling` to skip the VLM.
`parser: glm_ocr` also skips `vision:`, but scanned pages require a running
local Ollama `glm-ocr` model.

---

## Managed Qwen work

```yaml
collection: my_docs
summary_batch_size: 15
```

The managed local Qwen runtime supplies standard semantic query terms and
optional background entity, hierarchy, and demand-summary work. Fitz downloads
it on first model-backed operation, not during `point()`. A query-expansion
failure is traced and falls back to the literal plan; a background failure is
reported in enrichment status without invalidating the source index. Exact
model/runtime details live in [Managed Models](MANAGED_MODELS.md).

---

## Production: enterprise gateway with M2M + custom CA

```yaml
collection: production_docs
synthesizer: enterprise/openai/gpt-4o
query_intelligence: enterprise/openai/gpt-4o-mini
chat_base_url: https://llm.corp.internal/v1

auth:
  type: enterprise
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
# under <current-workspace>/.fitz/sqlite/. No DB knobs to configure.
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
from pathlib import Path

from fitz_sage.core import Query
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

cfg = FitzKragConfig(
    collection="my_docs",
    synthesizer=None,
    query_intelligence=None,
)
engine = FitzKragEngine(cfg)
engine.point(Path("./docs"), start_worker=False)
pack = engine.evidence(Query(text="What is quantum computing?"))
print(pack.mode, [item.file_path for item in pack.items])
```

Only `collection` is strictly required by the schema. `point()` completes the
searchable source index; the managed local Qwen runtime is loaded later by a
query or optional background enrichment.

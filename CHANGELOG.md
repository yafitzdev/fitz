# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 🎉 Highlights

**Multi-hop retrieval is on by default.** Retrieval is now one
`RetrievalPass` (retrieve → rerank → read) that the multi-hop controller
loops, gated by the pyrrho classifier instead of a per-hop chat call. A
single hop stays the common case, and the cross-encoder reranker now
runs on every query — multi-hop used to skip it.

**Dropped `optimum` → dropped `torch` (~2 GB).** The pyrrho governance
classifier and the gte-reranker cross-encoder now run on raw
`onnxruntime` instead of `optimum.onnxruntime`. `optimum 2.x`
hard-depends on `torch`, which the pre-built-INT8-ONNX load path never
needs — so `pip install fitz-sage` was dragging in ~2 GB of PyTorch
for nothing. Both encoders load the pre-quantized `.onnx` straight
from the model repo with `huggingface_hub` and run it through an
`onnxruntime.InferenceSession`. Verified torch-absent: pyrrho `decide()`
warm ~11 ms, reranker `rerank()` warm ~17 ms.

### 🔧 Fixed

- **Reranker now genuinely runs INT8 ONNX.** v0.13.0 loaded the
  reranker with `ORTModelForSequenceClassification.from_pretrained(
  model_id, export=True)` — which exports **FP32** ONNX from the
  PyTorch checkpoint on the fly, (a) ignored the pre-built quantized
  ONNX the model repo already ships, and (b) required `torch` to do
  the export (and `torch` was not a declared dependency — a fresh
  install + first rerank call could crash). The CHANGELOG + docs
  claimed "INT8 ONNX". Now `OnnxReranker` loads the pre-built
  `onnx/model_int8.onnx` (151 MB) that
  `Alibaba-NLP/gte-reranker-modernbert-base` publishes — genuine INT8.

### 🔄 Changed

- **`fitz_sage/governance/pyrrho.py`** + **`fitz_sage/llm/providers/onnx_reranker.py`** —
  rewrote both load paths: `huggingface_hub.hf_hub_download` to fetch
  the pre-built `.onnx`, `onnxruntime.InferenceSession` to run it,
  `transformers.AutoTokenizer` for tokenisation. No `optimum` import
  anywhere.
- **`OnnxReranker`** gained `onnx_subfolder` / `onnx_file` constructor
  params so a custom `model_id` can point at wherever that repo keeps
  its pre-built ONNX. There is no longer a torch-backed `export=True`
  fallback — if the named ONNX can't be fetched, `_load()` raises a
  clear error (pick a model that ships ONNX).
- **Dependencies:** dropped `optimum[onnxruntime]`; added `onnxruntime`
  and `huggingface-hub` as explicit direct deps. `transformers` stays
  (tokeniser only — torch is an optional extra of transformers and is
  no longer pulled). `TRANSFORMERS_VERBOSITY=error` is set before the
  tokenizer import to silence the benign "no DL framework found"
  advisory.
- **Extracted `OnnxEncoderBackend`** (`fitz_sage/encoders/onnx.py`) — a
  shared base class for the local INT8 ONNX encoders. `pyrrho.py` and
  `onnx_reranker.py` were duplicating the lock-guarded lazy load,
  `hf_hub_download` + `InferenceSession` setup, and tokenizer wiring
  line for line; that path now lives once in `OnnxEncoderBackend`.
  `Pyrrho` and `OnnxReranker` are thin subclasses that own only their
  tokenizer call shape and logit post-processing. `OnnxReranker`'s
  constructor + public surface are unchanged.
- **`enable_guardrails` → `governance` (provider-presence config).** The
  `enable_guardrails: bool` flag is replaced by `governance: <spec> | null`,
  matching the `rerank` / `vision` / `parser` pattern. `governance: pyrrho`
  (default) enables the classifier; `governance: null` disables it;
  `governance: pyrrho/<hf-model-id>` selects a custom pyrrho fine-tune.
  A config still using `enable_guardrails` gets an actionable error from
  the loader pointing at the replacement. The engine builds the classifier
  through a new `governance.create_governance()` factory; the module-level
  `governance.decide()` wrapper + its process-wide singleton are gone — the
  engine owns a config-built `Pyrrho` instance.
- **Retrieval stack unified into one `RetrievalPass`.** Tiers 1-4 of
  retrieval — candidate generation, cross-strategy fusion, precision
  rerank, read — were duplicated: the engine ran them inline for
  single-hop queries while `KragHopController` ran its own copy (minus
  reranking) for multi-hop. They are now one `RetrievalPass` unit
  (`engines/fitz_krag/retrieval/retrieval_pass.py`) — the engine runs it
  once, the hop controller loops it. The cross-encoder reranker
  consequently runs on **every** query, multi-hop included (multi-hop
  previously bypassed it).
- **Multi-hop is on by default** (`enable_multi_hop: true`) and free of
  the per-hop sufficiency chat call. `KragHopController` no longer asks a
  chat model "is this enough?" after each hop — it reads the pyrrho
  verdict: `TRUSTWORTHY`/`DISPUTED` -> stop, `ABSTAIN` -> bridge question
  + another pass. A single hop stays the common case; the added cost on
  the default path is one ~30 ms ONNX pyrrho call, no chat call.
- **Removed the dead `multi_hop` query signal.** The LLM-detected
  `multi_hop` extended signal (in `query_batcher`'s prompt and
  `RetrievalProfile`) never affected routing and is redundant now that
  pyrrho gates hops — dropped from both.

---

## [0.13.0] - 2026-05-15

### 🎉 Highlights

**Two encoder swaps. The retrieval + governance pipeline is now
fully local CPU.**

1. **Governance is [pyrrho](https://huggingface.co/yafitzdev/pyrrho-modernbert-base-v1)
   now.** The constraint+sklearn cascade is gone. Every
   `(query, retrieved contexts)` pair runs through a single INT8 ONNX
   forward pass over the ModernBERT-base pyrrho fine-tune and returns
   one of `TRUSTWORTHY` / `DISPUTED` / `ABSTAIN` in ~30 ms on CPU.
2. **Reranking is an ONNX cross-encoder now.** The chat-call-based
   `LLMReranker` is gone. Reranking runs locally via
   [`Alibaba-NLP/gte-reranker-modernbert-base`](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) —
   a 149M-parameter ModernBERT cross-encoder served as INT8 ONNX
   through `optimum.onnxruntime`. ~30–100 ms CPU for 10–20 candidates.

Both encoders share the same `optimum`/`transformers`/INT8 ONNX
toolchain. After this release the chat endpoint is only used for
query rewriting, multi-query decomposition, detection, and the
synthesizer.

**Governance swap (pyrrho replaces the cascade):**

| Metric                  | Cascade (v0.12.x) | Pyrrho v1 (v0.13.0) | Δ        |
| ----------------------- | ----------------- | ------------------- | -------- |
| Overall accuracy        | 78.7%             | **86.13%**          | +7.43 pp |
| False-trustworthy rate  | 5.7%              | **5.27%**           | -0.43 pp |
| Wall-clock per decision | ~500–2000 ms      | **~30 ms**          | ~50x     |
| External LLM calls      | 4–5               | **0**               | —        |

**Reranker swap (ONNX cross-encoder replaces `LLMReranker`):**

| Metric                  | LLMReranker (v0.12.x) | OnnxReranker (v0.13.0)              |
| ----------------------- | --------------------- | ----------------------------------- |
| Per-query reranker cost | 1 chat call           | 1 ONNX forward pass                 |
| CPU latency (10–20 docs)| ~500–2000 ms          | ~30–100 ms                          |
| External dependency     | chat endpoint         | none (local, cached)                |
| Ranking quality         | model-dependent       | matches 1.2B `nemotron-rerank` on Hit@1 |

Net code change in `fitz_sage/`: **-6,254 lines** (-12% of production
code). Combined with v0.12.0 we're at **-20,805 lines vs v0.11.0**
(-30%).

### 🚀 Added

- **`fitz_sage/governance/pyrrho.py`** — single-pass governance
  classifier. `decide(query, contexts) -> GovernanceDecision`,
  lazy-loaded INT8 ONNX, calibrated `TAU = 0.50` fallback on
  low-confidence `TRUSTWORTHY` predictions.
- **`fitz_sage/llm/providers/onnx_reranker.py`** — `OnnxReranker`.
  Lazy-loads tokenizer + INT8 ONNX `SequenceClassification` model via
  `optimum.onnxruntime`. Batched forward over `(query, doc)` pairs;
  returns `RerankResult` list sorted by score desc. Accepts any HF
  cross-encoder via `OnnxReranker(model_id=...)`.
- New rerank spec **`onnx`** (default) and **`onnx/<hf-model-id>`** in
  `create_rerank_provider`. Examples:
  - `rerank: onnx` → `gte-reranker-modernbert-base`
  - `rerank: onnx/BAAI/bge-reranker-base`
  - `rerank: onnx/jinaai/jina-reranker-v3`
  - `rerank: onnx/cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Runtime deps:** `transformers>=4.50`, `optimum[onnxruntime]>=1.20`,
  `numpy>=1.26` (numpy was implicit via downstream deps; now explicit).

### 🗑 Removed

**Governance cascade (-6,316 lines):**

- `fitz_sage/governance/decider.py` — `GovernanceDecider` and the 4-question cascade orchestrator
- `fitz_sage/governance/governor.py` — `AnswerGovernor`, `decide_answer_mode`
- `fitz_sage/governance/constraints/` (entire subtree):
  - `plugins/conflict_aware.py`, `insufficient_evidence.py`, `causal_attribution.py`, `specific_info_type.py`, `answer_verification.py`
  - `semantic.py` (`SemanticMatcher` + the unused embedder slot)
  - `feature_extractor.py` (108-feature extraction over chunks + chat outputs)
  - `aspect_classifier.py`, `numerical_detector.py`, `staged.py`, `runner.py`, `base.py`
- `fitz_sage/governance/data/model_v6_cascade.joblib` — the trained sklearn cascade artifact
- `tools/governance/` — feature extraction + cascade training scripts
- `CONSTRAINT_REGISTRY` and the constraint plugin discovery path from `fitz_sage/core/registry.py`
- Constraint plugin generator template + plugin type (`fitz_sage/plugin_gen/templates/constraint/`, `PluginType.CONSTRAINT`)
- Tests for the removed surface: `test_governance.py`, `test_governance_decider.py`, `test_constraints.py`, `test_constraint_runner.py`, `test_staged_pipeline.py`, `test_causal_attribution.py`, `test_krag_guardrails.py`, `tests/integration/test_governance_{constraints,pipeline}.py`

**LLM reranker:**

- `fitz_sage/llm/providers/llm_reranker.py` — `LLMReranker` and
  the chat-call-based reranker. Its tests (`tests/unit/llm/test_llm_reranker.py`)
  were removed alongside it.
- `"llm"` rerank spec — no longer accepted. The engine-layer
  shortcut in `FitzKragEngine` that special-cased `"llm"` is gone;
  all rerank providers now flow through `create_rerank_provider`.

### 🔄 Changed

- **`FitzKragConfig.enable_guardrails`** still toggles governance; default unchanged (`True`). When `True`, the engine calls `pyrrho.decide()` between retrieval and generation. When `False`, all answers are `TRUSTWORTHY`.
- **`FitzKragConfig.rerank` default**: `"llm"` → `"onnx"`. Existing configs that explicitly set `rerank: "llm"` will fail to load with an actionable error message. Migration: change to `rerank: onnx`.
- **Default config** (`fitz_krag/config/default.yaml`) reflects the new rerank default + updated comments.
- **`fitz_sage/llm/providers/__init__.py`** — exports `OnnxReranker` (was `LLMReranker`). Optional import so static tooling on a fresh checkout still works without `optimum` installed.
- **`HierarchyEnricher`** dropped its `semantic_matcher` parameter (was unused since v0.12.0). Same for `assess_chunk_group(chunks)` — no more `semantic_matcher=` kwarg.
- **`engine.py`** simplification: `_build_conflict_context` removed; `DISPUTED` cases now feed the synthesizer a single-key reason dict from `GovernanceDecision.reason`. ABSTAIN-mode gap context unchanged. Reranker setup dropped the two-branch shim — one code path now: `get_reranker(spec)`.
- **Docs swept end-to-end**: `CONSTRAINTS.md` (rewritten around pyrrho), `CONFIG.md`, `CONFIG_EXAMPLES.md`, `FEATURE_CONTROL.md`, `PLUGINS.md`, `ARCHITECTURE.md`, `ENGINES.md`, `features/platform/krag.md`, `features/platform/openai-compatible-endpoint.md`, `features/platform/enterprise-gateway.md`, `features/retrieval/reranking.md` (rewritten), `sparse-search.md`, `multi-query-rag.md`, `features/ingestion/{code-symbol-extraction, hierarchical-rag}.md`, `evaluation/beir-results.md`. Governance-benchmarking doc marked historical. README architecture diagram now shows local CPU encoders alongside the chat provider.

### 🔧 Fixed

- `fitz_sage/core/registry.py`: removed broken `CONSTRAINT_REGISTRY` reference + `get_constraint_plugin` / `available_constraint_plugins` accessors. The constraint plugin scan path (`fitz_sage.governance.constraints.plugins`) was already dead after deletion; removing it tidies up the registry.

### Migration from v0.12.x

```yaml
# old
rerank: llm

# new (default — faster + local)
rerank: onnx

# or pin a different cross-encoder
rerank: onnx/BAAI/bge-reranker-base
```

No code changes required for consumers of `FitzKragEngine` or the
public `get_reranker` API — the `RerankProvider` protocol is
unchanged. First call to `rerank()` and `decide()` downloads the
respective HF model (cached under `~/.cache/huggingface/`) and
lazy-loads it via `optimum.onnxruntime`.

---

## [0.12.0] - 2026-05-14

### 🎉 Highlights

**Storage swap** — PostgreSQL is gone; SQLite + FTS5 is the only
storage. One `.db` file per collection, zero install, stdlib only.
Drops `psycopg`, `psycopg-pool`, `fitz-pgserver` from deps. The
865-line `PostgresConnectionManager` becomes a ~200-line file-based
`SqliteConnectionManager`.

**Embeddings removed.** The entire embedding pipeline — provider
protocol, retrieval-side `embed_batch` calls, HyDE, contextual
embeddings, chunk-fallback, vector columns — is gone. Retrieval is
now BM25 (native FTS5 `bm25()`) + KRAG typed-unit routing + an LLM
reranker that scores documents in a single chat call. No vector DB,
no embedding model, no second network protocol. Validated on
fitz-gov v5 — the chat-only stack matches or beats the prior hybrid
on the metrics that matter.

**Single chat protocol.** OpenAI-compatible HTTP is the only chat
path. `endpoint` is the canonical provider; `openai`, `azure_openai`,
`enterprise` are presets over it. Legacy `cohere`, `anthropic`,
`ollama` provider names removed (raise `ValueError` with migration
text). Ollama still works — point `chat_base_url` at
`http://localhost:11434/v1`.

Smoke baseline preserved: 5/5 answered, 6/6 substrings, 3/5
governance mode-match.

### 🗑 Removed

- **`fitz_sage/storage/postgres.py`** — 865-line PostgreSQL connection manager (`8fa36a57`)
- **`fitz_sage/cloud/`** + **`fitz_sage/integrations/`** — Fitz Cloud cache client + LangChain/LlamaIndex adapter wrappers (`c4ab1138`)
- **`fitz_sage/evaluation/`** — governance decision logger + BEIR/RGB/fitz_gov benchmarks (postgres-coupled) (`8fa36a57`)
- **`fitz_sage/cli/commands/eval.py`** + **`reset.py`** — `fitz eval` and pgserver reset commands (`8fa36a57`, `c4ab1138`)
- **`fitz_sage/retrieval/sparse/`** — dead code that queried a `chunks` table deleted in the vector_db demolition (`8fa36a57`)
- **`fitz_sage/vector_db/`** + every plugin in it — vector-DB abstraction and pgvector backend (`bdca80ae`)
- **Embedding API** — `OpenAICompatEmbedding`, the `Embedder` provider protocol, `get_embedder()`, all per-provider embedding implementations, ingestion-side `embed_batch` calls, and the `embedding:` config key (`c7dc48a1`, `23fb8606`, `39e6ac02`, `b936b45d`)
- **HyDE** (`fitz_sage/retrieval/hyde/`) — hypothetical document generation; pointless without embeddings (`65ad6962`)
- **ChunkFallbackStrategy** — fell back to dense chunk search when typed-unit retrieval missed; obsolete in chat-only mode (`65ad6962`)
- **Contextual embeddings module** — summary-prefixed embeddings for disambiguation (`c7dc48a1`)
- **DiffIngestExecutor** surface — the embedding-aware ingest orchestrator; KRAG's own ingest pipeline took over (`65ad6962`)
- **Legacy chat providers** — `cohere`, `anthropic`, `ollama` per-vendor implementations. Passing these names raises `ValueError` with migration text pointing at `endpoint` + `chat_base_url` (`c7dc48a1`)
- **`FitzKragConfig.cloud` field** + engine `_cloud_client` / `_check_cloud_cache` / `_store_cloud_cache` / `_build_cache_versions` / `CLOUD_OPTIMIZER_VERSION` (`c4ab1138`)
- **`retrieval_mode` config flag** — was the transitional toggle between hybrid and chat-only modes; chat-only is now the only mode (`b936b45d`)
- **Postgres-specific tests:** `test_postgres_{connection,recovery,table_store}.py`, `test_evaluation.py`, `test_beir_benchmark.py`, `tests/integration/test_cloud_cache_e2e.py`, `tests/unit/integrations/`, `tests/manual/` (`8fa36a57`, `c4ab1138`)
- **Deps:** `psycopg`, `psycopg-pool`, `fitz-pgserver` (storage); `pgvector`, `faiss-cpu` (vector DB); `langchain-core`, `llama-index-core` extras (`8fa36a57`, `c4ab1138`, `bdca80ae`)

### 🔄 Changed

- **Storage backend** — PostgreSQL → SQLite. New `SqliteConnectionManager` with WAL mode + `foreign_keys=ON`; one `.db` per collection under `<workspace>/sqlite/` (`8fa36a57`)
- **Retrieval pipeline** — embeddings + RRF over dense/sparse → pure FTS5 `bm25()` + KRAG typed-unit routing + LLM reranker. The reranker speaks the same OpenAI-compatible chat protocol as the synthesizer; no separate cross-encoder backend (`b936b45d`, `c7dc48a1`)
- **Chat protocol** — per-vendor SDK paths consolidated into `OpenAICompatChat`. `endpoint` is canonical; `openai`, `azure_openai`, `enterprise` are configuration presets. Per-tier overrides (`chat_smart_base_url`, `chat_smart_model`, `chat_smart_api_key_env`) let you mix local and cloud (`c7dc48a1`, `969bfb43`)
- **CLI overrides** — `fitz query --endpoint <URL> --model <name> --api-key-env <VAR>` lets users point at any OpenAI-compatible server without editing config (`969bfb43`)
- **Retrieval ranking** — FTS5 `MATCH` + native `bm25()` replaces `tsvector @@ to_tsquery` + `ts_rank` (`8fa36a57`)
- **Schema port** — `JSONB` → `TEXT` + JSON1; `TEXT[]` → JSON arrays + `json_each`; `ILIKE` → `LIKE COLLATE NOCASE`; `unnest(columns)` → `json_each(columns)`; `%s` → `?` (`8fa36a57`)
- **Collection lifecycle** — `list_collections` scans `glob fitz_*.db`; `delete_collection` is `os.unlink` (was `DROP DATABASE` against admin DB) (`8fa36a57`)
- **`PostgresTableStore` → `SqliteTableStore`** — dynamic per-CSV table model preserved; LLM SQL prompts updated to SQLite syntax (`8fa36a57`)
- Stores ported to SQLite: `RawFileStore`, `SymbolStore`, `SectionStore`, `ImportGraphStore`, `TableStore`, `VocabularyStore`, `EntityGraphStore` (`8fa36a57`)

### 🔧 Fixed

- **`SqliteConnectionManager` deadlock**: class-level `_lock` was a non-reentrant `threading.Lock`, so `reset_instance()` → `stop()` (which also takes the lock) deadlocked from the same thread. Switched to `threading.RLock`.
- **Unit-test singleton cascade**: the ~25 failures in `test_vocabulary` / `test_krag_guardrails` / `test_section_store` / `test_krag_engine` were caused by `test_krag_detection` and `test_krag_engine` patching `SqliteConnectionManager` without resetting the singleton afterward; the MagicMock leaked into later tests. Both files now apply the opt-in `reset_sqlite_singleton` fixture via module-level `pytestmark = pytest.mark.usefixtures(...)`.
- **`test_krag_engine` stale-postgres patch**: `PostgresTableStore` import path no longer exists post-Cloud-removal; the engine constructs `SqliteTableStore` instead. Updated the patch target.
- **`test_section_store` BM25 sign**: the test fed a positive raw `bm25()` value but production code negates the FTS5 result (FTS5 returns lower-better; downstream wants higher-better). Test input flipped to negative so the assertion matches reality.
- **`test_cli_endpoint_flags` ANSI brittleness**: linux CI runners render typer/rich help with embedded color escapes that split `--endpoint` into `-` + `-endpoint`, breaking the substring assertion. Test now strips ANSI before matching.
- **CI workflow stale-postgres steps**: `.github/workflows/ci.yml` had a "Run postgres tests (Linux only)" step that selected zero tests post-Cloud-removal (exit code 5). Removed it, dropped the now-pointless `-m "not postgres"` filter, and pruned `fitz-pgserver` / `psycopg` / `psycopg-pool` / `faiss-cpu` from the pip install lists. `mutation.yml` dropped its `--ignore=tests/unit/test_postgres_recovery.py` for a file that no longer exists.

---

## [0.11.0] - 2026-03-21

### 🎉 Highlights

**Local/Cloud Optimization** — Local Ollama automatically maps all chat tiers to `chat_balanced` (one model, zero VRAM swaps). Cloud providers use all three tiers as configured (no swap cost with APIs).

**Hybrid PDF Parser** — Replaced Docling (21 min for 113 pages) with pdfplumber + GLM-OCR hybrid parser (28s). Text pages parsed instantly via pdfplumber with font-size/bold heading detection; scanned pages routed to GLM-OCR.

**Retrieval Benchmarks** — Document retrieval eval (20 queries, 3 PDFs, 75% critical recall) and table row retrieval eval (20 queries, 3 CSVs) with automated scoring against ground truth.

### 🚀 Added

- **`QueryBatcher`** — batches analysis + detection into 1 LLM call, halving model-swap overhead (`9e09793`)
- **`RetrievalProfile`** — single dataclass unifying 3 fragmented retrieval trigger mechanisms (analysis weights, detection flags, config constants) (`a4a5943`)
- **Extended classification signals** — specificity, domain, answer_type, multi_hop as soft multipliers on retrieval behavior (`777e7fc`)
- **Hybrid PDF parser** (`glm_ocr.py`) — pdfplumber fast path + GLM-OCR fallback for scanned/image pages (`62c8ff7`)
- **Phase 1 content embedding** — embed `title + content[:2000]` immediately during parsing, skip 25-min LLM summary wait (`9552705`)
- **Document retrieval benchmark** (`doc_eval.py`) — 20 queries across IRS 1040, NIST AI RMF, RAG survey PDFs (`5069470`)
- **Table retrieval benchmark** (`table_eval.py`) — 20 queries testing full pipeline: table discovery → SQL generation → execution (`29cf080`)
- Roadmap doc for query intelligence pipeline (`9e09793`)

### 🚀 Performance

- **Local/cloud tier split** — local ollama: all tiers → `chat_balanced` (zero VRAM swaps); cloud: use configured fast/balanced/smart (`8130ddd`)
- **Parallel embed + classify** — embed runs in parallel with batch classify since embed model + chat model both fit in VRAM (`a73a22b`)
- **Single chat tier on local** — map fast/balanced/smart all to `chat_balanced`, eliminating 5 model swaps per query on Ollama (`a7904d2`)
- **Combined SQL generation** — merged column selection + SQL gen into 1 LLM call in TableQueryHandler (`a7904d2`)
- **Rewrite-first dispatch** — rewrite query before classification for better analysis accuracy (`9e09793`)
- **Eliminated double SQL execution** — table handler reuses validated SQL result instead of executing twice (`3f483ba`)

### 🔄 Changed

- HyDE ownership moved to router only — removed from code_search and section_search strategies (`5069470`)
- Router `retrieve()` accepts `RetrievalProfile` instead of separate `analysis`/`detection` params (`a4a5943`)
- Deleted 4 static gating methods from router (`_should_run_hyde`, `_should_run_multi_query`, `_should_inject_corpus_summaries`, `_should_run_agentic`) — logic moved to `RetrievalProfile` (`a4a5943`)
- `parser` config option now supports `"glm_ocr"` in addition to `"docling"` and `"docling_vision"` (`62c8ff7`)
- TableQueryHandler uses single LLM call for SQL generation (removed separate column selection step) (`a7904d2`)

### 🔧 Fixed

- SQL prompt: added GROUP BY rule for aggregate queries (`29cf080`)
- SQL retry: feed actual PostgreSQL error messages to LLM instead of generic "Query execution failed" (`29cf080`)
- SQL prompt: added rule to include ORDER BY/WHERE columns in SELECT (`29cf080`)
- Removed dead `_text_to_elements` and `_parse_json_list` methods (`3f483ba`)

### 🧪 Tests

- Rewrote performance/load tests to measure harness overhead, not hardware-dependent LLM speed
- Removed absolute latency/throughput thresholds — tests now check relative metrics (degradation ratio, memory stability)
- Reduced tier 4 answer() calls from 96 to 36 (~9 min instead of 60+ min)
- Updated chaos tests to mock synthesizer chat (pipeline no longer calls `_query_analyzer.analyze`)

### 📝 Docs

- Rewrote CONFIG.md for flat `provider/model` config format
- Updated 12 docs: removed `plugin_name`/`kwargs`, `fitz_krag.yaml`, `local_ollama`, `fitz init` references

---

## [0.10.4] - 2026-03-19

### 🔄 Changed

- **Removed LM Studio provider** — Ollama is the only local LLM provider. Simplifies the provider stack. fitz-graveyard has its own independent LM Studio implementation. (`2b07944`)
- Removed LM Studio from firstrun fallback chain, README, architecture diagram, docs (`2b07944`, `e5cde2d`)

### 🔧 Fixed

- **Single-file agentic search always uses the file** — when user points at exactly one file with `--source`, skip BM25 filtering and use it directly (`7b24a93`)

---

## [0.10.3] - 2026-03-19

### 🎉 Highlights

**Flat Config** — Single config file (`.fitz/config.yaml`) with flat `provider/model` keys. No nested `chat_kwargs`, no engine-specific config directory. `chat_fast: ollama/qwen3.5:0.6b` is the entire config for a chat tier. Auto-created on first run.

**Zero-Friction First Run** — `pip install fitz-sage` then `fitz query "Q" --source ./docs` just works. Auto-detects Ollama models, classifies into tiers, writes config. If models are missing, prompts to pull them. Fallback chain: Ollama → LM Studio → API keys → clear instructions.

**Lightweight Install** — `docling` moved to optional extra (`pip install fitz-sage[docs]`). Base install includes lightweight PDF/DOCX/PPTX parsers via pypdfium2, python-docx, python-pptx (~25MB instead of ~5GB).

**Simplified CLI** — Removed `fitz init`, `fitz config`, `fitz eval` from public CLI. Config is auto-created and users edit `.fitz/config.yaml` directly. Four commands remain: `query`, `collections`, `serve`, `reset`.

### 🚀 Added

- Flat config schema: `chat_fast`, `chat_balanced`, `chat_smart` replace `chat` + `chat_kwargs.models` (`37b6832`)
- First-run auto-detection: Ollama model discovery via `/api/tags`, tier classification by parameter size (`2c328d5`)
- Interactive model pull prompt when Ollama has no suitable models (`2c328d5`)
- Fallback chain: Ollama → LM Studio → Cohere/OpenAI API keys → clear error (`2c328d5`)
- "Ollama installed but not running" detection via PATH binary check (`666fc87`)
- Lightweight PDF parser (pypdfium2) with heading heuristics (`a6c0063`)
- Lightweight DOCX parser (python-docx) with structural parsing (`a6c0063`)
- Lightweight PPTX parser (python-pptx) with slide/title extraction (`a6c0063`)
- FAQ/Troubleshooting section in README (`4e4340c`)
- Config file guard on `fitz serve` — refuses to start without config, gives instructions (`25dadd7`)
- Progress messages during engine init: "Starting database...", "Loading LLM models..." (`a8e6c71`)
- Actionable Ollama errors: 404 → "run `ollama pull X`", ConnectError → "run `ollama serve`" (`2c328d5`, `7f6df38`)
- `pytest_sessionstart` cleanup for zombie postgres processes on Windows (`0b3a95c`)

### 🔄 Changed

- Config: single file `.fitz/config.yaml` replaces `.fitz/config.yaml` + `.fitz/config/fitz_krag.yaml` (`37b6832`)
- Config: `embedding` default includes model (`ollama/nomic-embed-text`) (`37b6832`)
- Deleted `chat_kwargs`, `embedding_kwargs`, `rerank_kwargs`, `vision_kwargs` from schema (`37b6832`)
- Deleted `FitzPaths.engine_config()`, `config_dir()`, `ensure_config_dir()` (`37b6832`)
- `get_chat_factory()` accepts tier specs dict instead of single provider string (`37b6832`)
- SDK `_ensure_config()` uses first-run auto-detection instead of hardcoded Cohere template (`910c576`)
- `docling` moved from core dependency to `[docs]` extra (`a8e6c71`)
- Parser router falls back to lightweight parsers when docling not installed (`a6c0063`)
- Removed `fitz init`, `fitz config`, `fitz eval` CLI commands (`c60a57e`, `8c328c6`)
- All "run fitz init" messages replaced with "edit .fitz/config.yaml" (`c60a57e`)
- README: added prereqs line, FAQ section, removed inline fallback notes (`666fc87`, `4e4340c`)

### 🔧 Fixed

- Clean error message when Ollama not running (was raw WinError 10061 stacktrace) (`7f6df38`)
- Warning when PDF/DOCX files encountered without docling installed (`a8e6c71`)

### 📝 Docs

- Removed all `fitz init`/`fitz config`/`fitz eval` references from CLI.md, CONFIG.md, PLUGINS.md, FEATURE_CONTROL.md, TROUBLESHOOTING.md (`b905c8c`)
- README: LM Studio added as prerequisite option (`cd9cdc6`)
- README: CLI reference reduced to 4 commands with config note (`c60a57e`)
- FAQ covers: fitz not found, PDF support, Ollama errors, model changes, cloud providers, reset (`4e4340c`)

---

## [0.10.2] - 2026-03-12

### 🎉 Highlights

**Standalone Code Retrieval (`fitz-sage[code]`)** — New `fitz_sage/code/` module provides LLM-powered code retrieval without PostgreSQL, pgvector, or docling. Point at a directory, ask a question — CodeRetriever builds a structural index from AST, selects relevant files via LLM, expands via import graph and neighbor directories, and returns compressed results. Zero heavy dependencies.

**LlmCodeSearchStrategy Overhaul** — Rewrote the DB-backed code search strategy: FILE-level addresses instead of per-symbol, combined query expansion + file selection in one LLM call (better targeting), import graph expansion, neighbor directory expansion, and flat origin-based scoring (1.0/0.9/0.8). The combined prompt produces more targeted file selections by letting the LLM reason about expansion terms and files holistically.

**LM Studio Provider** — New `lmstudio` chat provider with multi-tier model support. Configure different models for fast/balanced/smart tiers via YAML.

**Actionable Governance Modes** — ABSTAIN and DISPUTED answers are now informative and solution-oriented instead of generic refusals. ABSTAIN explains what was searched, shows related topics that DO exist, and suggests documents to add. DISPUTED tells the LLM exactly which sources conflict so it can explain both perspectives specifically.

### 🚀 Added

- `fitz_sage/code/` standalone module: `CodeRetriever`, `indexer`, `prompts` (`f4ec70b`)
- `CodeRetriever` class: index → LLM select → import expand → neighbor expand → read → compress pipeline (`f4ec70b`)
- `build_file_list()`, `build_structural_index()`, `build_import_graph()` in `fitz_sage/code/indexer.py` (`f4ec70b`)
- `get_file_paths()` and `get_structural_index()` public accessors on `CodeRetriever` (`1ab902c`)
- Configurable `llm_tier` parameter on `CodeRetriever` — consumers choose which model tier does file selection (`010de5a`)
- `[code]` extras group in pyproject.toml for dependency documentation (`f4ec70b`)
- LM Studio chat provider with tier-based model selection (`d3ff5ce`)
- 20 unit tests for code retrieval (indexer, retriever, import graph, no-heavy-imports) (`f4ec70b`)
- **Actionable ABSTAIN** — ABSTAIN answers now explain why (governance reasons), show related corpus topics (via entity graph), and suggest what documents to add (`2196861`)
- **Actionable DISPUTED** — DISPUTED mode injects specific conflicting excerpts and source names into the LLM prompt so it explains both perspectives (`0ae9dc5`)
- `EntityGraphStore.find_related_topics()` for corpus gap analysis (`2196861`)
- `FitzKragEngine._build_gap_context()` and `_build_conflict_context()` for governance intelligence surfacing (`2196861`, `0ae9dc5`)
- Corpus Intelligence and KRAG Agent roadmap documents (`2196861`)
- Foundation file detection — auto-include files with >10 reverse imports (protocols, data models, enums) in code retrieval (`356b182`)
- Hub protection — hub files, foundation files, and scan hits can't be displaced by post-limit facade swap (`356b182`)
- Query-aware hub import ranking — hub imports ranked by keyword overlap with search terms instead of competing equally (`79232b9`)
- Retrieval quality benchmark with 40-query ground truth across 10 categories (`2c6eaee`)
- Eval tooling: auto-load LM Studio models, `limit`/`max_manifest_chars` params, A/B test scripts (`84a4406`)

### 🔄 Changed

- `LlmCodeSearchStrategy` rewritten: FILE-level addresses, combined expand+select prompt, neighbor expansion, flat scoring (`3e7e827`)
- All YAML plugin references updated to Python provider terminology (`8fb0758`)
- Configuration schemas unified with base classes (`b655466`)
- Structured logging with context tracking throughout codebase (`6418339`)
- `CodeSynthesizer.generate()` accepts `gap_context` and `conflict_context` for actionable governance messages (`2196861`, `0ae9dc5`)
- Early "no addresses" ABSTAIN now sets `Answer.mode = AnswerMode.ABSTAIN` properly (`2196861`)
- `ConflictAwareConstraint` stores conflicting chunk excerpts in constraint metadata (`0ae9dc5`)
- Priority ordering updated: selected > hub core > foundation > hub imports > facade > import > neighbor (`caa4629`)
- Foundation files ranked by query keyword overlap, same as hub imports (`caa4629`)
- SDK: removed `ask()` alias — `query()` is the only method on the `fitz` class (`5de811b`)

### ⚡ Performance

- Combined query expansion + file selection in one LLM call (was two separate calls) — better targeting with fewer API calls (`3e7e827`)
- AST-based structural index with connection-weighted truncation — important files keep detail under budget (`f4ec70b`)
- Python compression via `compress_python()` reduces context size before LLM processing (`f4ec70b`)

### 🔧 Fixed

- Major technical debt cleanup across codebase (`8d42a57`)
- `__version__` synced to `0.10.2` (was stuck at `0.10.1`) (`5de811b`)
- Markdown file path extraction fallback when LLM skips JSON format — fixed 5/40 eval queries (`28bd121`)
- Eval uses `provider/model` spec format for chat factory (`9fcab3a`)
- Eval uses absolute paths to avoid PyCharm working directory issues (`fb7ff33`)
- Stale config test defaults updated to match schema evolution (lmstudio, top_addresses=50, top_read=50, max_context_tokens=48000)
- Heavy imports test runs in subprocess to avoid `fitz_pgserver` contamination from other tests

### 📝 Docs

- Updated all plugin references from YAML to Python providers (`8fb0758`)
- Synced documentation with v0.10.1 changes (`e8d8f97`)
- Doc audit: fixed API.md (`provenance` → `sources`), CONFIG.md (nested config → boolean flags), CONTRIBUTING.md (removed nonexistent `[ingest]` extra) (`5de811b`)
- README rewritten: hallucination before/after hero, "How is this different from LangChain" section, actionable failures bullet (`2d30dd3`)

---

## [0.10.1] - 2026-02-28

### 🎉 Highlights

**ML Detection Classifier** — New `DetectionClassifier` combines lightweight ML models with keyword heuristics to gate expensive LLM detection calls. Temporal detection at 90.6% recall, comparison at 90.2% recall. Integrated into `DetectionOrchestrator` with full training pipeline.

**Retrieval Quality Overhaul** — Replaced asymmetric merge with proper Reciprocal Rank Fusion (k=60), removed min_relevance_score filter that was killing recall, and tuned HNSW ef_search=200 for better vector search accuracy.

**SemanticMatcher Unification** — Consolidated semantic classification under a single `SemanticMatcher` abstraction. Migrated `CausalAttribution` and `ConflictAware` detectors to use it.

### 🚀 Added

- ML+keyword `DetectionClassifier` with training script and model artifacts (`981b6e9`, `cc9b8c6`)
- `DetectionClassifier` integrated into `DetectionOrchestrator` for smart gating (`2909e07`, `debacfa`)
- 32 unit tests for DetectionClassifier and orchestrator gating (`521905c`)
- Hybrid BM25+semantic retrieval wired into BEIR benchmark (`5fba5e3`)
- Widened entity graph expansion with corpus summary injection for thematic queries (`f1df503`)
- BEIR benchmark results and methodology docs (`ef0ec73`, `aedfc21`)
- Confirmed fiqa score for bge-m3: 0.2702 (`4a94ca1`)

### ⚡ Performance

- Proper Reciprocal Rank Fusion (k=60) replacing asymmetric merge (`6c8f988`)
- Removed `min_relevance_score` filter that was killing recall (`d3ae3b0`)
- Set `hnsw.ef_search=200` for vector search queries (`c4924a9`)

### 🔄 Changed

- Unified semantic classification under `SemanticMatcher` (`0821132`)
- Migrated `CausalAttribution` and `ConflictAware` to `SemanticMatcher` (`9c7c081`)
- Schema-driven feature extraction for governance classifier (`7abbcfe`)
- 5-fold CV for cascade classifier with safety-calibrated thresholds (`71ae026`)

### 🔧 Fixed

- Added `num_ctx` support to `OllamaEmbedding` for context window control (`65da2eb`)
- Updated tests for SemanticMatcher-backed constraint detection (`b94383f`)

---

## [0.10.0] - 2026-02-17

### 🎉 Highlights

**Progressive KRAG with Agentic Search** — Query any file or folder instantly without pre-ingestion. `fitz query --source ./docs "question"` parses documents on-demand, indexes in the background, and serves answers immediately. Agentic search discovers relevant files from a manifest, parses only what's needed, and retrieves with full KRAG intelligence.

**4-Question Cascade Governance Classifier** — Replaced the two-stage ML classifier with a 4-question cascade architecture: Q1 (evidence sufficient? ML) → Q2 (conflict? rule: ca_fired) → Q3 (conflict resolved? ML) → Q4 (evidence solid? ML). Achieves 79.1% accuracy with 90.0% abstain recall and 76.2% disputed recall. Model now ships with `pip install fitz-sage`.

**40% Pipeline Speedup** — Smart retrieval gating skips unnecessary LLM calls for simple queries, overlapped embedding fetches dimensions during component init, parallel strategy execution, and pre-warmed LLM/embedding models eliminate cold-start latency.

**CLI Simplification** — Slimmed CLI from 14 commands to 7. Consolidated `fitz point` and `fitz quickstart` into `fitz query --source`. Cleaner, more discoverable command surface.

### 🚀 Added

#### Progressive KRAG & Agentic Search
- `fitz query --source <path>` — Query files/folders without pre-ingestion
- On-demand PDF/DOCX/PPTX parsing with background indexing
- Agentic search: manifest-based file discovery, selective parsing, KRAG retrieval
- Pipeline timing breakdown in query output (parse, retrieve, generate times)
- Cached parsed PDF text to avoid redundant parsing
- Heading structure cache for rich documents (eliminates double parsing)

#### Governance Classifier Improvements
- 4-question cascade classifier (Q1→Q2→Q3→Q4) replacing two-stage architecture
- Text answer features: `query_subject_partial`, `entity_substantive_score`, `best_sentence_coverage`, `best_span_length`, `answer_span_coverage`
- Conflict quality features: `conflict_to_number_ratio`, `opposing_conclusion_count`, `negation_per_char`, `short_ctx_with_overlap`
- Interaction features: `ix_av_fires_good_overlap`, `ix_max_div_per_conflict`, `ix_single_chunk_denial`, `ix_ie_no_ca`, `ix_ca_no_ie`
- ~21 missing governance features added, InsufficientEvidence constraint re-enabled
- Numerical divergence features for cross-chunk analysis
- Safety-focused threshold calibration with vectorized sweep
- Model artifact shipped with package (`fitz_sage/governance/data/model_v6_cascade.joblib`)

#### Retrieval Intelligence
- Retrieval intelligence fully wired through KRAG pipeline
- Task-type embedding prefixes for improved retrieval quality
- Rewritten AV jury constraint: 3-fast + balanced confirmation

#### Performance Optimizations
- Smart retrieval gating: skip detection/analysis for simple queries
- Overlapped embedding: fetch `embed.dimensions` during component init (-1s startup)
- Parallel strategy execution with shared embeddings and pgserver sharing
- Pre-warm LLM and embedding models during engine init
- Skip analysis LLM call for simple queries (heuristic classification)
- Run query rewrite in parallel with analysis+detection
- Skip LLM selection for small manifests
- Warm smart tier sequentially after fast tier during init

#### Code Extraction Robustness
- Regex fallback for Python files with syntax errors (AST parse fails gracefully)
- Regex fallback for TypeScript/Java/Go when tree-sitter is unavailable

### 🔄 Changed

- **CLI surface**: Slimmed from 14 commands to 7 — removed `point`, `quickstart`, `chunk`, `db`, `engine`, `plugin`, `collections`
- **`fitz query --source`**: Consolidates `fitz point` and `fitz quickstart` into single command
- **Governance classifier**: Two-stage RF→ET replaced by 4-question cascade (Q1=0.62, Q3=0.56, Q4=0.51)
- **Governance data**: 199 "trustworthy-with-gap" cases relabeled to abstain in fitz-gov benchmark (context genuinely doesn't answer the question)
- **GovernanceDecider**: Model loaded exclusively from package directory (`fitz_sage/governance/data/`)

### 🔧 Fixed

- **InsufficientEvidence constraint**: Embedder was always `None` — now correctly passed at init
- **IE embedder API**: Fixed to use `.embed()` method instead of calling embedder directly
- **IE false ABSTAIN**: Fixed false abstain for lowercase proper nouns
- **Agentic search over-retrieval**: Fixed excessive chunk retrieval in single-chunk scenarios
- **Single-file source handling**: Fixed stale manifest accumulation
- **PDF content reading**: Fixed content extraction and suppressed noisy Docling/RapidOCR logs
- **Manifest management**: Re-add unchanged files to manifest after clear
- **Parsed text cache**: Always ensure cache exists during registration
- **RICH_DOC_EXTENSIONS**: Fixed undefined reference in agentic search
- **PostgreSQL crash recovery**: Hardened recovery with better stale lock handling
- **pgserver pool exhaustion**: Prevent pgserver restart on `PoolTimeout`
- **GovernanceDecider wiring**: Fixed ML classifier integration, punctuation bug, and defaults
- **Guardrails tier**: Use fast tier for guardrails, fix cold start warmup
- **Calibration safety**: Vectorized sweep with safe abstain fallback
- **Null chat response**: Handle gracefully instead of crashing

### 🧪 Testing

- Test suite overhaul: deleted stale tests, added E2E format coverage, added unit tests
- Fixed 26 test failures from PostgreSQL crash recovery hardening

### 🧹 Housekeeping

- Removed old governance models (v1-v7), eval results, and analysis scripts (~100MB freed)
- `*.joblib` added to package-data for model shipping

---

## [0.9.0] - 2026-02-12

### 🎉 Highlights

**KRAG Engine — Sole Engine Architecture** — The Knowledge Routing Augmented Generation (KRAG) engine replaces `fitz_rag` as the only engine. KRAG introduces multi-strategy query routing (code, section, table, chunk), multi-language code extraction (Python, TypeScript, Java, Go), and address-based retrieval with full expansion (references, imports, section context). The `fitz_rag` engine has been deleted entirely — no shims, no compatibility layer.

**Multi-Strategy Query Routing** — Queries are classified by `QueryAnalyzer` into types (code, documentation, data, cross, general) and routed to specialized retrieval strategies with weighted scoring. Each strategy searches a different index (symbol index, section index, table store, chunk store) and results are merged and ranked.

**Retrieval Robustness** — New `min_relevance_score` config field (default 0.15) filters out low-relevance results from vector search, preventing nonsense queries from polluting LLM context. Strategy calls are wrapped in try-except for graceful handling of missing tables on cloud tiers. Connection pool lifecycle management prevents pool exhaustion across tier switches.

### 🚀 Added

#### KRAG Engine (`fitz_sage/engines/fitz_krag/`)
- `FitzKragEngine` — Full `KnowledgeEngine` implementation with multi-strategy retrieval
- `QueryAnalyzer` — LLM-based query classification into code/documentation/data/cross/general types
- `QueryAnalysis` — Frozen dataclass with `strategy_weights` for weighted retrieval routing
- `RetrievalRouter` — Dispatches to code, section, table, chunk strategies and merges results
- `AddressExpander` — Expands retrieved addresses with references, imports, and section context
- `KRAGPipeline` — Full pipeline orchestration (analyze → route → expand → generate)

#### Multi-Language Code Extraction
- `PythonExtractor` — AST-based extraction of classes, functions, imports with relative import resolution
- `TypeScriptExtractor` — TypeScript/JavaScript class, function, interface extraction
- `JavaExtractor` — Java class, method, interface extraction
- `GoExtractor` — Go struct, function, interface extraction
- Symbol index for code-aware retrieval across all supported languages

#### Retrieval Robustness
- `min_relevance_score` config field — Filters addresses below threshold after ranking
- Graceful strategy failure handling — Missing tables/indices log warnings instead of crashing
- `PostgresConnectionManager.close_pool()` — Explicit pool cleanup to prevent connection exhaustion
- Pool cleanup on tier switch in e2e test runner

#### OllamaVision Provider
- `OllamaVision` — Local VLM provider for figure description during ingestion
- VLM parsing integration in KRAG ingestion pipeline

#### Guardrails & Governance Integration
- Guardrails (conflict-aware, insufficient-evidence, causal-attribution) integrated into KRAG
- Cloud cache integration for KRAG pipeline
- Shared detection system (`DetectionOrchestrator`) integrated into KRAG retrieval

### 🔄 Changed

- **Sole engine**: `fitz_krag` is now the only engine — all CLI, runtime, SDK, and API paths updated
- **`fitz init`**: Engine selection removed; KRAG plugin selection integrated
- **DATA query weights**: Rebalanced from `{table: 0.85, section: 0.05}` to `{table: 0.70, section: 0.15}` to prevent document-content queries from being misrouted to table strategy
- **DATA classification prompt**: Sharpened to distinguish explicit tabular operations from questions about facts/specifications that happen to involve numbers
- **pgvector dimension detection**: Now uses `format_type(atttypid, atttypmod)` returning strings like `"vector(384)"` instead of raw integer dimension queries, preventing dimension mismatch across embedding model changes
- **Governance thresholds**: Tuned to s1=0.55, s2=0.79 (15 critical cases)
- **Two-stage classifier**: Added support in eval pipeline
- **Test suite**: Security, chaos, load, and performance tests migrated from fitz_rag to KRAG API

### 🗑️ Removed

#### fitz_rag Engine (replaced by fitz_krag)
- `fitz_sage/engines/fitz_rag/` — Entire engine directory deleted
- `fitz_sage/engines/fitz_rag/retrieval/` — RAG pipeline, steps, strategies
- `fitz_sage/engines/fitz_rag/generation/` — RGS answer generation
- `fitz_sage/engines/fitz_rag/config/` — Engine configuration
- All `fitz_rag` imports, references, and CLI assumptions removed across codebase

#### Other Removals
- `LLMError` compatibility shim — Direct imports only
- `BasePluginConfig` / `PluginKwargs` duplicates — Extracted to `core/config.py`
- Governance guardrails moved from `core/` to shared `fitz_sage/governance/`

### 🔧 Fixed

- **Vector dimension mismatch**: pgvector now detects and prevents dimension mismatches when switching embedding models on an existing collection
- **Query misrouting**: Document-content queries (battery sizes, CEO names) no longer misclassified as DATA queries
- **CSV NULL handling**: Empty CSV cells stored as SQL NULL instead of empty string, fixing `IS NULL` queries
- **Connection pool exhaustion**: Pools explicitly closed when switching tiers, preventing PoolTimeout errors
- **EngineRegistry global state**: Integration tests no longer pollute the global engine registry
- **Nonsense query handling**: Low-relevance results filtered out, preventing irrelevant context from reaching the LLM

### 📚 Documentation

- Updated README governance numbers to current v3.0 results
- Rewritten v3.0 evaluation docs to reflect 3-class ML classifier
- Updated fitz-gov category references after qualification/confidence rename
- Governance benchmarking docs updated with production numbers
- Research notepad restructured to prevent LLM taxonomy confusion

### 📦 Migration from 0.8.x

This is a **breaking release**. The `fitz_rag` engine no longer exists.

- **Import paths**: Replace all `fitz_sage.engines.fitz_rag` imports with `fitz_sage.engines.fitz_krag`
- **Engine name**: Replace `engine="fitz_rag"` with `engine="fitz_krag"` in all API/SDK calls
- **Config files**: Engine config is now at `engines/fitz_krag/config/default.yaml`
- **No compatibility layer**: There are no shims or deprecation warnings — update all references

---

## [0.8.1] - 2026-02-09

### 🎉 Highlights

**ML Governance Classifier** — Replaced the hand-coded `AnswerGovernor` (37% accuracy) with a two-stage ML classifier: Random Forest (answerable vs abstain) → Extra Trees (trustworthy vs disputed). Trained on 1,113 fitz-gov cases with 51 features. Per-class recall: Abstain 81.2%, Disputed 89.7%, Trustworthy 70.6%. Only 3 dangerous (disputed→trustworthy) errors in 1,100+ cases.

**GovernanceDecider Integration** — New `GovernanceDecider` class wraps the ML classifier with fail-open fallback to `AnswerGovernor`. Loads the model artifact once at init and runs two-stage prediction with calibrated per-class thresholds (s1=0.50, s2=0.785).

**Safety-First Threshold Tuning** — Iterative threshold exploration prioritizing dispute detection safety. Sweet-spot at s2=0.785 balances trustworthy recall (70.6%) against disputed recall (89.7%) with minimal dangerous misclassifications.

### 🚀 Added

#### Governance ML Classifier
- Two-stage classifier pipeline: RF (Stage 1: answerability) → ET (Stage 2: conflict detection)
- `GovernanceDecider` class with fail-open fallback to `AnswerGovernor` on any error
- Calibrated per-class thresholds (s1=0.50, s2=0.785) tuned for safety
- 3-class output (abstain/disputed/trustworthy) mapped to 4-class AnswerMode (ABSTAIN/DISPUTED/CONFIDENT/QUALIFIED)
- Feature extraction pipeline: 51 features from constraint results, chunk metadata, and inter-chunk text signals
- Inter-chunk text features (hedging ratio, negation ratio, numeric density) — +10.5pp Stage 2 CV improvement
- Feature parity fix: `ctx_*` features ported from training to production inference

#### Evaluation & Experiments
- 10+ classifier experiments documented (Exp 1–10) with full result tracking
- Per-class calibrated thresholds with governor fallback (Step 1)
- Two-stage binary classifier formalization with calibration (82.96% accuracy, 76.9% min recall)
- Expanded dataset evaluation (1,113 cases from fitz-gov 3.0)
- Dead code audit identifying 18 removable features and 700+ lines of dead code

#### Testing
- 90 new vector_db unit tests covering types, writer, loader, custom plugin, and registry
- Property-based tests for vector_db components

### 🔧 Fixed

- Feature parity gap between training and production inference (`ctx_*` features missing at inference time)
- Governance constraint sensitivity tuning (causal attribution false positives, IE forecast-year relaxation)
- Evidence character gate for ConflictAware constraint
- Primary referent abstain rule for InsufficientEvidence constraint

### 📚 Documentation

- Updated README governance section with current classifier results and two-stage pipeline diagram
- Research notepad with full experiment history and threshold tuning journal
- Classifier status notepad tracking model iterations
- Source agreement features analysis (blocked by single-source test set, deferred to fitz-gov v4.0)
- fitz-gov 3.0 docs, cross-check fixes, governance journey writeup
- Dead code audit results and calibration analysis

### 🧹 Refactoring

- Removed 18 dead features, 2 unused plugins, 600+ lines of dead code
- Cleaned up governance constraint plugins (SIT moved to Stage 2, rate info type added)
- Removed scratch benchmark script from repo

---

## [0.8.0] - 2026-02-03

### 🎉 Highlights

**fitz-gov Benchmark Integration** - New governance-focused evaluation benchmark using the fitz-gov package. Enables systematic testing of epistemic governance constraints (conflict detection, causal attribution, insufficient evidence) across 6 categories with two-pass LLM validation.

**Enhanced Governance Constraints** - Improved accuracy and observability for all governance constraint plugins with semantic relevance checks and better integration with the RAG pipeline.

### 🚀 Added

#### Benchmarking & Evaluation
- fitz-gov benchmark integration using external `fitz-gov` package
- Two-pass LLM validation for governance constraint accuracy
- Support for 6 governance categories: conflict awareness, causal attribution, insufficient evidence, qualification, dispute, and semantic relevance
- CLI display for all 6 evaluation categories
- Integration tests for governance constraints

#### Governance & Constraints
- Semantic relevance checking in governance constraints
- Improved governance analyzer accuracy with better chunk handling
- Enhanced conflict-aware, causal-attribution, and insufficient-evidence plugins
- Governance-only evaluation mode (skips answer quality categories)
- Better observability for governance constraint violations

### 🔧 Fixed

- fitz-gov loader to support new data structure from GitHub releases
- Chunk instantiation in benchmark evaluations (pass Chunk objects instead of dicts)
- RGSAnswer attribute access (use `answer` instead of `text`)
- Import paths for constraints and governance modules
- Engine configuration schema handling in benchmarks

### 📚 Documentation

- Multiple documentation updates and clarifications
- Step-by-step guides for governance constraint usage
- Benchmark evaluation examples

### 🧹 Refactoring

- Refactored FitzGovBenchmark to use external fitz-gov package
- Simplified benchmark structure to focus on governance validation
- Removed metadata assignment from RGSAnswer for cleaner separation
- Better pipeline component integration

---

## [0.7.1] - 2026-02-01

### 🎉 Highlights

**Enterprise Authentication System** - New enterprise-grade auth framework with dynamic token refresh, mTLS support, and circuit breaker patterns for production resilience. Supports OAuth2, API key rotation, and composite multi-header authentication.

**Reranking Intelligence** - Reranking is now baked directly into the dense vector search plugin with smart skip logic when no rerank provider is configured. Seamless integration without separate configuration flags.

**Multi-Dimension Cloud Cache** - Cloud cache API now supports multiple embedding dimensions, enabling projects with different embedding models to share the same cache infrastructure.

### 🚀 Added

#### Enterprise Authentication (`fitz_sage/llm/auth/`)
- `DynamicHttpxAuth` - Dynamic token refresh with callback support for httpx clients
- `TokenProviderAdapter` - Adapter pattern for OAuth2/API key providers
- `CompositeAuth` - Multi-header authentication for complex scenarios
- `M2MAuth` enhancements - Added retry logic with exponential backoff and circuit breaker
- `EnterpriseAuth` - New auth type for enterprise gateway providers
- Certificate validation utilities with expiry checking and chain validation
- mTLS support across all LLM providers (Anthropic, OpenAI, Cohere)

#### Reranking
- Baked reranking into `DenseVectorSearchStep` with automatic skip when `rerank: null`
- Smart provider detection - only runs rerank when provider is configured
- No configuration flags needed - presence of rerank provider IS the toggle

#### Cloud Cache
- Multi-dimension embedding support in cache key generation
- Dimension validation and compatibility checking
- Graceful handling of single-dimension cache entries during migration

#### Testing Infrastructure
- Property-based tests with Hypothesis for vocabulary variations
- Mutation testing with mutmut (weekly CI + local overnight)
- E2E integration tests for cloud cache
- pgserver recovery tests (unit + integration)
- Test tier markers for granular test execution

### 🔧 Fixed

- pgserver auto-recovery on Windows with improved stale lock handling
- Mutation testing CI workflow output parsing for mutmut 2.x
- Integration support for multiple embedding dimensions
- Cloud cache edge cases with dimension mismatches

### 📚 Documentation

- Updated embedding dimension references across docs
- Added enterprise auth examples
- Improved troubleshooting guides

### 🧪 Testing

- Added comprehensive test coverage for dynamic auth, circuit breakers, and retry logic
- Certificate validation test suite
- Auth provider integration tests
- Property-based testing for vocabulary variations

---

## [0.7.0] - 2026-01-26

### 🎉 Highlights

**Unified PostgreSQL Storage** - Replaced FAISS + SQLite with PostgreSQL + pgvector for all storage needs. Uses `pgserver` (pip-installable embedded PostgreSQL) for local mode with zero external dependencies. One database per collection, automatic schema management, and HNSW indexing for fast vector search.

**Native PostgreSQL Tables** - Tabular data (CSV/tables) now stored directly in PostgreSQL with automatic schema inference. Enables SQL queries over structured data alongside vector search.

**LLM Factory Pattern** - New `chat_factory()` function provides clean LLM client instantiation with proper dependency injection. Replaces scattered client creation logic.

### 🚀 Added

#### PostgreSQL Storage System (`fitz_sage/storage/`)
- `PostgresConnectionManager` - Singleton connection manager with pgserver lifecycle
- `StorageConfig` - Pydantic config for local/external PostgreSQL modes
- Per-collection database isolation (one DB per collection)
- Automatic pgvector extension initialization
- Connection pooling via `psycopg_pool`
- pgserver graceful shutdown with file handle cleanup on Windows
- Auto-recovery on corrupted data directories

#### pgvector Backend (`fitz_sage/backends/local_vector_db/pgvector.py`)
- `PgVectorDB` - Full VectorDBPlugin implementation
- HNSW indexing with configurable `m` and `ef_construction`
- Hybrid search combining vector similarity + full-text search (tsvector)
- Native PostgreSQL `tsvector` for sparse/BM25-style retrieval
- `scroll()` and `scroll_with_vectors()` for batch iteration
- Automatic schema creation on first use

#### PostgreSQL Table Store (`fitz_sage/tabular/store/postgres.py`)
- `PostgresTableStore` - Native PostgreSQL storage for tabular data
- Gzip-compressed CSV storage in BYTEA column
- Hash-based deduplication
- Automatic schema inference and column tracking

#### LLM Factory (`fitz_sage/llm/chat/factory.py`)
- `chat_factory()` - Clean factory function for chat client instantiation
- Proper dependency injection pattern
- Tier-based model selection (smart/fast)

#### Test Infrastructure
- Test tier markers: `tier1` (unit), `tier2` (integration), `tier3` (e2e), `tier4` (performance)
- `pytest.ini` configuration for tier-based test execution
- pgserver test fixtures with auto-cleanup
- Windows-specific file handle cleanup in tests

### 🔄 Changed

- **Default vector DB**: `faiss` → `pgvector` in default config
- **Vocabulary storage**: YAML files → PostgreSQL `keywords` table
- **Sparse index**: BM25 files → PostgreSQL `tsvector` column (auto-maintained)
- **Entity graph**: SQLite files → PostgreSQL `entities` + `entity_chunks` tables
- **Table store**: SQLite/generic → PostgreSQL native tables
- **Collection delete**: Now drops entire PostgreSQL database (auto-cleans all related data)

### 🗑️ Removed

#### Legacy Storage Backends
- `fitz_sage/backends/local_vector_db/faiss.py` - Replaced by pgvector
- `fitz_sage/vector_db/plugins/local_faiss.yaml` - Replaced by pgvector.yaml
- `fitz_sage/tabular/store/sqlite.py` - Replaced by postgres.py
- `fitz_sage/tabular/store/generic.py` - Replaced by postgres.py
- `fitz_sage/tabular/store/qdrant.py` - Replaced by postgres.py
- `fitz_sage/tabular/store/cache.py` - No longer needed

#### Knowledge Map Module
- `fitz_sage/map/` - Experimental module removed (not production-ready)
- `fitz map` CLI command removed

#### Deprecated Path Helpers
- `vocabulary()` path function - Now emits deprecation warning
- `sparse_index()` path function - Now emits deprecation warning
- `entity_graph()` path function - Now emits deprecation warning

### 📦 Dependencies

New:
```toml
"psycopg[binary]>=3.1"
"psycopg-pool>=3.1"
"pgvector>=0.2.0"
"pgserver>=0.1.0"
```

Removed:
```toml
"faiss-cpu>=1.7.0"  # Now optional via [faiss] extra
```

### 🧪 Testing

- 198 tier1 tests passing
- 62 postgres-specific tests
- 67 vocabulary tests (migrated to PostgreSQL)
- Windows-compatible pgserver tests with file handle cleanup

---

## [0.6.2] - 2026-01-24

### 🎉 Highlights

**Unified LLM-Based Query Classification** - Consolidated all scattered query detection systems (temporal, aggregation, comparison, freshness, expansion) into a single unified detection module. One LLM call now classifies all query intents instead of separate regex-based detectors. More accurate classification with lower latency.

### 🚀 Added

#### Unified Detection System (`fitz_sage/retrieval/detection/`)
- `LLMClassifier` - Combines all detection modules into one LLM call
- `DetectionOrchestrator` - Unified registry with `DetectionSummary` result
- `DetectionModule` ABC - Modular detection with `prompt_fragment()` and `parse_result()`
- Detection modules: `TemporalModule`, `AggregationModule`, `ComparisonModule`, `FreshnessModule`, `RewriterModule`
- `ExpansionDetector` - Dict-based synonym/acronym expansion (non-LLM, fast)
- `DetectionResult` dataclass with confidence scores and metadata
- `DetectionCategory` enum for type-safe detection types

### 🔄 Changed

- **Query classification is now LLM-based** - More accurate than regex patterns, handles edge cases better
- **VectorSearchStep** - Now uses `DetectionOrchestrator` for all query classification
- **Retrieval strategies** - Updated to receive `DetectionResult` instead of legacy detector outputs

### 🗑️ Removed

#### Legacy Detection Systems (consolidated into unified detection)
- `fitz_sage/retrieval/aggregation/` - Replaced by `detection/modules/aggregation.py`
- `fitz_sage/retrieval/temporal/` - Replaced by `detection/modules/temporal.py`
- `fitz_sage/retrieval/expansion/` - Replaced by `detection/detectors/expansion.py`
- `fitz_sage/engines/fitz_rag/retrieval/steps/freshness.py` - Replaced by `detection/modules/freshness.py`

### 📚 Documentation

- Updated feature docs to reflect unified detection system
- CLAUDE.md already documented the new detection architecture

---

## [0.6.1] - 2026-01-23

### 🎉 Highlights

**HyDE (Hypothetical Document Embeddings)** - Generate hypothetical document passages that would answer abstract queries, then search with both original and hypothetical embeddings. Bridges the semantic gap between conceptual questions and concrete document content. Queries like "What's their approach to sustainability?" now find relevant EV/battery/emissions docs.

**Contextual Embeddings** - Chunks are now embedded with their summaries prepended, providing richer semantic context. This resolves pronoun ambiguity and improves retrieval quality for chunks that reference concepts without naming them explicitly. Inspired by Anthropic's Contextual Retrieval technique.

**Query Rewriting** - LLM-powered query rewriting resolves conversational context (pronouns, references), fixes typos, removes filler words, and optimizes queries for document retrieval. Enables natural multi-turn conversations with proper context resolution.

**Conversational Context for SDK & API** - The SDK and REST API now support passing conversation history for context-aware retrieval. Queries like "tell me more about it" now work correctly in programmatic use cases.

### 🚀 Added

#### HyDE - Hypothetical Document Embeddings (`fitz_sage/retrieval/hyde/`)
- `HypothesisGenerator` - Generates 3 hypothetical document passages per query
- Single fast-tier LLM call for all hypotheses
- Hypotheses embedded and searched alongside original query
- RRF (Reciprocal Rank Fusion) merges results from all searches
- Always-on when chat client is available (no configuration needed)
- Graceful degradation on LLM failure
- `prompts/hypothesis.txt` - Externalized prompt template
- Documentation: `docs/features/hyde.md`
- E2E test scenarios for HyDE validation

#### Contextual Embeddings (`fitz_sage/ingestion/`)
- Summary-prefixed embedding: chunks are embedded as `f"{summary}\n\n{content}"` instead of just content
- Zero additional LLM calls - uses summaries already generated by enrichment pipeline
- Graceful fallback when no summary exists
- Implemented in both `IngestionPipeline` and `DiffIngestExecutor`
- Documentation: `docs/features/contextual-embeddings.md`

#### Query Rewriting (`fitz_sage/retrieval/rewriter/`)
- `QueryRewriter` - LLM-powered query transformation with conversation context
- `RewriteResult` - Structured result with rewritten query, confidence, and reasoning
- `ConversationContext` - Typed conversation history for pronoun resolution
- Rewrite types: conversational (pronouns), clarity (typos), retrieval (optimization)
- Ambiguity detection with multi-query expansion
- Single fast-tier LLM call per query (~100-200ms overhead)
- Graceful degradation on LLM failure (uses original query)
- `prompts/rewrite.txt` - Externalized prompt template
- Documentation: `docs/features/query-rewriting.md`
- Comprehensive test suite: `tests/unit/test_rewriter.py` (469 lines)

#### Conversational Context for SDK & API
- `fitz_sage/sdk/fitz.py` - `query()` now accepts `conversation_history` parameter
- `fitz_sage/api/routes/query.py` - POST `/query` accepts `conversation_history` in request body
- `fitz_sage/api/models/schemas.py` - Added conversation history to query schema
- Enables context-aware retrieval in programmatic use cases

#### Small Chunk Enrichment Skipper
- Skip LLM enrichment for chunks below minimum token threshold
- Configurable threshold in enrichment config
- Saves costs on small/trivial chunks

### 🔄 Changed

- `VectorSearchStep` now integrates query rewriter for conversational context resolution
- `FitzRagEngine.answer()` accepts optional conversation history
- Chat command passes conversation history to retrieval pipeline

### 🗑️ Removed

#### Dead Code Cleanup
- `fitz_sage/ingestion/enrichment/cache.py` - Unused summary cache (replaced by content-hash in state)
- `fitz_sage/ingestion/enrichment/router.py` - Unused enrichment router
- `fitz_sage/ingestion/enrichment/base.py` - Unused base class
- `fitz_sage/ingestion/enrichment/python_context.py` - Unused Python-specific context builder

### 🐛 Fixed

- Security tests now use proper mock fixtures
- Load/scalability tests fixed for CI stability
- Performance tests fixed with proper conftest setup
- Input validation tests corrected
- Rewriter prompt formatting fixes
- HyDE strategy integration fixes
- Various test fixture and configuration fixes

---

## [0.6.0] - 2026-01-21

### 🎉 Highlights

**Structured Data Module** - Complete rewrite of structured/tabular data handling with SQL generation, derived fields, schema detection, and unified vector+structured query routing. Enables natural language queries over CSV/tables with automatic SQL generation and result formatting.

**Fitz Cloud Integration** - Full integration with Fitz Cloud for query-time RAG optimization. Supports encrypted cache lookup/storage, model routing, and retrieval fingerprinting for cache keys.

**VLM Figure Description** - Docling parser now supports Vision Language Model (VLM) integration for automatic figure/chart description. When configured with a vision provider, images detected in documents are described by the VLM instead of showing "[Figure]" placeholders.

**Direct Text Ingestion** - Ingest text directly from command line without files: `fitz ingest "Your text here"`. Auto-detects text vs file paths.

**Architecture Overhaul** - Major refactoring to eliminate anti-patterns: typed models replace `dict[str, Any]`, god classes split into focused modules, global state eliminated, and Protocol-based type hints throughout. Test consolidation following DHH principles.

### 🚀 Added

#### Structured Data Module (`fitz_sage/structured/`)
- `schema.py` - Schema detection and field type inference (458 lines)
- `sql_generator.py` - Natural language to SQL translation (415 lines)
- `executor.py` - Safe SQL execution with sandboxing (404 lines)
- `derived.py` - Derived field computation (ratios, aggregates) (438 lines)
- `router.py` - Intelligent query routing (vector vs structured) (239 lines)
- `formatter.py` - Result formatting for LLM consumption (199 lines)
- `ingestion.py` - CSV/table ingestion with schema extraction (305 lines)
- `types.py` - Type definitions and protocols (369 lines)
- `constants.py` - SQL templates and constants (64 lines)
- `fitz tables` CLI command for table management (525 lines)
- Structured E2E test suite (989 lines)
- Vector search integration for derived fields

#### Direct Text Ingestion (`fitz_sage/cli/commands/ingest_direct.py`)
- `ingest_direct_text()` - Ingest text strings directly
- `is_direct_text()` - Auto-detect text vs file path
- `fitz ingest "Your text here"` - CLI support
- Automatic doc_id generation

#### Fitz Cloud (`fitz_sage/cloud/`)
- `CloudClient` - HTTP client for Fitz Cloud API
- Query-time RAG optimizer integration
- Model routing from cloud configuration
- Encrypted cache lookup and storage in RAGPipeline
- `retrieval_fingerprint` for deterministic cache keys
- `X-API-Key` header authentication

#### VLM Figure Description (`fitz_sage/ingestion/parser/plugins/docling.py`)
- `_describe_image_with_vlm()` - Sends detected figures to VLM for description
- `generate_picture_images=True` option in Docling pipeline
- PIL image extraction via `item.get_image(doc)`
- VLM call statistics tracking (`vlm_calls`, `vlm_errors`)
- 300s timeout for VLM calls (model loading on first call)

#### Docling Grid-Based Table Extraction
- `_build_table_from_grid()` - Extracts clean markdown tables from Docling's structured grid data
- Bypasses `export_to_markdown()` which adds unwanted bold formatting
- Proper column normalization and separator generation

#### Framework Integrations
- LangChain retriever abstraction layer

#### Figure E2E Tests (`tests/e2e/`)
- `FIGURE_RETRIEVAL` feature type in scenarios
- `figure_test.pdf` fixture with embedded bar chart
- 4 new scenarios (E145-E148) for figure content retrieval

### 🔄 Changed

#### Architecture Refactoring
- **Typed Models** - Replaced `dict[str, Any]` anti-pattern with proper dataclasses and typed models throughout codebase
- **Split `ingest.py`** (1,005 lines) into focused modules under `cli/commands/ingest/`
- **Split `init.py`** (1,033 lines) into focused modules under `cli/commands/init/`
- **Split `FitzPaths`** god class - Eliminated global state mutations
- **Protocol Type Hints** - Added Protocol-based type hints for documentation
- **`PluginKwargs`** - Typed class replacing `**kwargs` anti-pattern
- **Exception Handling** - Consolidated repeated exception handling in RAGPipeline
- **ClaraEngine** - Eliminated global state pollution

#### CLI Services Extraction
- `cli/services/ingest_service.py` - Extracted ingestion orchestration logic (319 lines)
- `cli/services/init_service.py` - Extracted initialization logic (275 lines)
- Clean separation of CLI presentation from business logic

#### Test Consolidation (DHH-style)
- Consolidated 12 granular RGS tests into `test_rgs_consolidated.py` (189 lines)
- Consolidated 6 context pipeline tests into `test_context_pipeline_consolidated.py` (106 lines)
- Removed ~300 lines of fragmented test files
- Each test file now tests a complete behavior, not implementation details

#### API Improvements
- Extracted API error decorator for consistent error handling
- Simplified tier resolution logic
- Added constants for magic values

#### Documentation
- `docs/api_reference.md` - New comprehensive API reference (233 lines)

#### Enrichment System
- E2E test rework for enrichment validation

### 🗑️ Removed

#### Consolidated Test Files (DHH-style cleanup)
- `test_context_pipeline_cross_file_dedupe.py`
- `test_context_pipeline_markdown_integrity.py`
- `test_context_pipeline_ordering.py`
- `test_context_pipeline_pack_boundary.py`
- `test_context_pipeline_unknown_group.py`
- `test_context_pipeline_weird_inputs.py`
- `test_rgs_chunk_id_fallback.py`
- `test_rgs_chunk_limit.py`
- `test_rgs_exclude_query.py`
- `test_rgs_max_chunks_limit.py`
- `test_rgs_metadata_format.py`
- `test_rgs_metadata_truncation.py`
- `test_rgs_no_citations.py`
- `test_rgs_prompt_core_logic.py`
- `test_rgs_prompt_slots.py`
- `test_rgs_strict_grounding_instruction.py`

### 🐛 Fixed

- **Table Markdown Formatting** - Tables no longer have bold headers (`**Column**`) that break downstream SQL generation
- **Security Test Assertion** - Fixed flaky security test assertion
- **Retrieval Latency Threshold** - Increased threshold for CI variance tolerance

### 📦 Configuration

#### VLM Configuration (`fitz_sage/llm/vision/local_ollama.yaml`)
- Increased endpoint timeout from 180s to 300s for VLM model loading
- Recommended model: `minicpm-v` for 16GB VRAM GPUs

---

## [0.5.2] - 2026-01-13

### 🎉 Highlights

**Multi-Hop Reasoning** - Fitz RAG now supports iterative multi-hop retrieval for complex queries requiring information synthesis across multiple documents. The system automatically detects when additional context is needed and performs follow-up retrievals.

**Entity Graph Expansion** - New entity graph system enriches retrieval by linking related chunks through shared entities. When a chunk mentions entities, the system automatically retrieves other chunks discussing the same concepts.

**Advanced Retrieval Intelligence** - Comprehensive suite of retrieval features now baked into the system including temporal queries, query expansion, hybrid search (dense+sparse), freshness/authority boosting, and aggregation query detection.

**End-to-End Testing Framework** - New comprehensive E2E test framework validates retrieval quality across diverse scenarios with automated validation and detailed reporting.

### 🚀 Added

#### Multi-Hop Reasoning (`fitz_sage/engines/fitz_rag/retrieval/multihop/`)
- `MultiHopController` - Orchestrates iterative retrieval with termination logic
- `InfoExtractor` - Extracts key information from intermediate results
- `CompletionEvaluator` - Determines when sufficient information has been gathered
- Configurable max hops and answer quality thresholds
- Multi-hop config in `FitzRagConfig` schema

#### Entity Graph System (`fitz_sage/ingestion/entity_graph/`)
- `EntityGraphStore` - Persistent storage for entity relationships
- Entity linking during ingestion enrichment
- Graph expansion step in retrieval pipeline
- Automatic retrieval of chunks sharing entities with top results
- Graph stored per collection in `.fitz/graphs/`

#### Retrieval Intelligence Suite (`fitz_sage/retrieval/`)
- **Temporal Queries** (`temporal/detector.py`) - Detects time-based comparisons and period filters
- **Query Expansion** (`expansion/expander.py`) - Generates synonym/acronym variations
- **Hybrid Search** (`sparse/index.py`) - BM25 sparse index with RRF fusion
- **Freshness & Authority** (`fitz_rag/retrieval/steps/freshness.py`) - Recency and authority boosting
- **Aggregation Queries** (`aggregation/detector.py`) - Detects statistical aggregation intent
- **Vocabulary System** (`vocabulary/`) - Exact keyword matching across chunks
  - `VocabularyDetector` - Extracts identifiers from content
  - `VocabularyMatcher` - Matches query terms to vocabulary
  - `VocabularyStore` - Persists keywords per collection
  - `VariationGenerator` - Generates term variations

#### End-to-End Testing (`tests/e2e/`)
- `E2ETestRunner` - Orchestrates full retrieval scenarios
- `TestReporter` - Generates detailed test reports
- `ScenarioValidator` - Validates retrieval results
- 15+ test scenarios covering:
  - Temporal queries (comparison, period filtering)
  - Sparse retrieval (exact keyword matching)
  - Tabular data routing
  - Conflict detection
  - Causal attribution
  - Code-aware search
  - Entity matching
- Test fixtures with structured markdown, code, CSV data

### 🔄 Changed

- **Module Organization**: Moved `vocabulary` and `entity_graph` modules from `fitz_sage/ingestion/` to `fitz_sage/retrieval/` for clearer separation of concerns
- **VectorSearchStep**: Now includes temporal handling, query expansion, hybrid search, multi-query expansion, and aggregation detection
- **EnrichmentPipeline**: Integrated entity graph construction during ingestion
- **Collection Delete**: Now cleans up entity graphs and vocabulary stores
- **README**: Major refactor with dedicated feature documentation pages in `docs/features/`
  - `aggregation-queries.md` - Statistical query handling
  - `code-aware-chunking.md` - Programming language support
  - `comparison-queries.md` - Entity comparison queries
  - `epistemic-honesty.md` - Constraint system
  - `freshness-authority.md` - Temporal relevance
  - `hierarchical-rag.md` - Multi-level summaries
  - `hybrid-search.md` - Dense + sparse fusion
  - `keyword-vocabulary.md` - Exact term matching
  - `multi-hop-reasoning.md` - Iterative retrieval
  - `multi-query-rag.md` - Query expansion
  - `query-expansion.md` - Synonym generation
  - `tabular-data-routing.md` - CSV/table handling
  - `temporal-queries.md` - Time-based filtering

### 🗑️ Removed

- `tools/smoketest/` - Replaced by E2E test framework
  - `smoke_fitz_rag_e2e.py` (750 lines)
  - `smoke_local_llm.py` (219 lines)

### 🐛 Fixed

- Tabular query handling now properly routes to registered tables
- Filesystem source plugin handles metadata more robustly
- Simple chunker improves overlap handling

---

## [0.5.1] - 2026-01-11

### 🎉 Highlights

**ChunkEnricher - Unified Enrichment Bus** - All chunk-level enrichment (summary, keywords, entities) is now baked in and runs automatically via a unified enrichment bus. The `ChunkEnricher` batches ~15 chunks per LLM call, making enrichment nearly free (~$0.13-0.74 for 1000 chunks).

**Exact Keyword Matching** - Keywords extracted during ingestion (test case IDs, ticket numbers, code identifiers) are now used for exact-match filtering at query time. Queries mentioning "TC-1001" will only return chunks containing that exact identifier.

**Multi-Query RAG** - Long or complex queries are automatically expanded into multiple focused search queries. 

**Comparison queries** - ("X vs Y") are detected and expanded to ensure both entities are retrieved.

**Table Registry** - CSV/table files are now reliably retrieved via a table registry that stores chunk IDs at ingestion time. No more missed tables due to low semantic similarity.

### 🚀 Added

#### ChunkEnricher (`fitz_sage/ingestion/enrichment/chunk/`)
- `ChunkEnricher` - Unified enrichment bus with extensible module architecture
- `EnrichmentModule` - Abstract base class for pluggable enrichment types
- `SummaryModule` - Generates searchable per-chunk summaries
- `KeywordModule` - Extracts exact-match identifiers (TC-1001, JIRA-123, `AuthService`)
- `EntityModule` - Extracts named entities (classes, people, technologies)
- Batched processing (~15 chunks per LLM call) for cost efficiency
- Keywords automatically saved to `VocabularyStore` for exact-match retrieval

#### Keyword Matching (`fitz_sage/engines/fitz_rag/retrieval/`)
- `KeywordMatcher` - Matches query terms against ingested vocabulary
- `VocabularyStore` - Persists auto-detected keywords per collection
- Keyword filtering in `VectorSearchStep` - filters results to chunks containing matched keywords

#### Multi-Query Expansion (`fitz_sage/engines/fitz_rag/retrieval/steps/vector_search.py`)
- Automatic query expansion for queries > 300 characters
- Comparison query detection (vs, compare, difference between)
- Comparison-aware expansion ensures both compared entities are retrieved
- Deduplication across expanded queries

#### Table Registry (`fitz_sage/tabular/registry.py`)
- `add_table_id()` / `get_table_ids()` - Store and retrieve table chunk IDs per collection
- Table IDs registered at ingestion time for reliable retrieval
- `retrieve()` method added to `VectorClient` protocol
- Table registry cleaned up on collection delete

### 🔄 Changed

- **Enrichment is now baked in**: Summary, keyword, and entity extraction run automatically when chat client is available
- **Removed opt-in config flags**: `enrichment.summary.enabled` and `enrichment.entities.enabled` removed
- **EnrichmentPipeline**: Now uses `ChunkEnricher` instead of separate summarizer and entity extractor
- **VectorClient protocol**: Added `retrieve(collection, ids)` method for direct ID-based lookup
- **Ingest UX**: Type a name to create new collection (no more "[0] + Create new" step)
- **Ingest UX**: Removed verbose "(docs corpus, hierarchical summaries)" and "VLM enabled" text
- **Documentation updated**: ENRICHMENT.md, INGESTION.md, CONFIG.md, ARCHITECTURE.md, README.md

### 🗑️ Removed

- `SummaryConfig` - No longer needed (summaries always on)
- `EntityConfig` - No longer needed (entities always on)
- `summaries_enabled` property on EnrichmentPipeline (replaced by `chunk_enrichment_enabled`)
- `entities_enabled` property on EnrichmentPipeline (replaced by `chunk_enrichment_enabled`)
- Dead code: `enabled_features` list in ingest command (was built but never displayed)

---

## [0.5.0] - 2026-01-07

### 🎉 Highlights

**Plugin Generator** - New `fitz plugin` command generates complete plugin scaffolds with templates, validation, and library context awareness. Generate chat, embedding, rerank, vision, chunker, retrieval, or constraint plugins with a single command.

**Parser Plugin System** - New parser abstraction replaces the reader module. Parsers handle document-to-structured-content conversion with plugins for plaintext, Docling (PDF/DOCX), and Docling+VLM (with figure description).

**Vision Plugin System** - Full YAML-based vision plugin support for VLM-powered figure description during ingestion. Supports Cohere, OpenAI, Anthropic, and Ollama vision models.

**Comprehensive Documentation** - Added 9 new documentation files covering API, architecture, configuration, constraints, enrichment, feature control, ingestion, SDK, and troubleshooting.

### 🚀 Added

#### Plugin Generator (`fitz_sage/plugin_gen/`)
- `fitz plugin generate` - Interactive plugin scaffolding wizard
- Template-based generation for all plugin types
- Library context awareness (detects installed packages)
- Validation and review workflow
- Templates for: `chunker`, `constraint`, `llm_chat`, `llm_embedding`, `llm_rerank`, `reader`, `retrieval`, `vector_db`

#### Parser Plugin System (`fitz_sage/ingestion/parser/`)
- `ParserRouter` - Routes files to appropriate parsers by extension
- `Parser` protocol with `can_parse()` and `parse()` methods
- `PlainTextParser` - Handles .txt, .md, .py, .json, etc.
- `DoclingParser` - PDF, DOCX, images via Docling library
- `DoclingVisionParser` - Docling + VLM for figure description
- `ParsedDocument` with typed `DocumentElement` structure

#### Vision Plugin System (`fitz_sage/llm/vision/`)
- YAML-based vision plugins matching chat/embedding pattern
- `cohere.yaml` - Cohere vision (command-a-vision-07-2025)
- `openai.yaml` - OpenAI vision (gpt-4o)
- `anthropic.yaml` - Anthropic vision (claude-sonnet-4)
- `local_ollama.yaml` - Ollama vision (llama3.2-vision)
- Vision plugin schema (`vision_plugin_schema.yaml`)
- Message transforms for vision requests

#### Source Abstraction (`fitz_sage/ingestion/source/`)
- `Source` protocol for file discovery
- `SourceFile` dataclass with URI, local path, metadata
- `FileSystemSource` plugin for local files

#### Documentation (`docs/`)
- `API.md` - REST API reference
- `ARCHITECTURE.md` - System design and layer dependencies
- `CONFIG.md` - Configuration reference
- `CONSTRAINTS.md` - Epistemic guardrails guide
- `ENRICHMENT.md` - Enrichment pipeline documentation
- `FEATURE_CONTROL.md` - Plugin-based feature control
- `INGESTION.md` - Ingestion pipeline guide
- `SDK.md` - Python SDK reference
- `TROUBLESHOOTING.md` - Common issues and solutions

#### CLI Improvements
- `fitz plugin` - New command for plugin generation
- `fitz init` - Vision model selection prompt added
- Vision provider/model configuration in init wizard

### 🔄 Changed

- **Parser replaces Reader**: `fitz_sage/ingestion/reader/` removed, replaced by `fitz_sage/ingestion/parser/`
- **Config schema**: `ExtensionChunkerConfig` now includes `parser` field for VLM control
- **Chunking router**: Now accepts parser selection via config
- **Init wizard**: Reordered sections (Chat → Embedding → Rerank → Vision → VectorDB)

### 🐛 Fixed

- `ParserRouter` no longer accepts invalid `vision_client` parameter
- Vision model defaults now use correct models (e.g., `command-a-vision-07-2025` not text model)
- Config validation accepts `parser` field in chunking config

### 🗑️ Removed

- `fitz_sage/ingestion/reader/` module (replaced by parser system)
- `fitz_sage/ingestion/chunking/engine.py` (consolidated into router)

---

## [0.4.5] - 2026-01-04

### 🎉 Highlights

**Zero-Friction Quickstart** - The `fitz quickstart` command now truly delivers on "zero-config RAG." Provider detection is fully automatic: Ollama detected → used; API key in environment → used; first time → guided through free Cohere signup. After initial setup, subsequent runs are completely prompt-free.

**CLIContext** - New centralized CLI context system provides a single source of truth for all configuration. Package defaults guarantee all values exist—no more scattered `.get()` fallbacks across commands.

**Collection Warnings** - The CLI now warns when a collection doesn't exist or is empty before querying, preventing confusing "I don't know" answers when the real issue is missing data.

### 🚀 Added

#### Zero-Friction Quickstart (`fitz_sage/cli/commands/quickstart.py`)
- **Auto-detection cascade**: Ollama → COHERE_API_KEY → OPENAI_API_KEY → guided signup
- `_resolve_provider()` - Detects best available LLM provider automatically
- `_check_ollama()` - Detects running Ollama with required models (llama3.2, nomic-embed-text)
- `_guide_cohere_signup()` - Step-by-step onboarding for new users (free tier)
- `_save_api_key_to_env()` - Cross-platform API key persistence (Windows: `.fitz/.env`, Unix: `.bashrc`/`.zshrc`)
- Removed engine selection prompt—quickstart now focuses on fitz_rag for simplicity

#### CLIContext (`fitz_sage/cli/context.py`)
- Centralized context for all CLI commands
- Guaranteed configuration values (package defaults always loaded)
- `get_collections()`, `require_collections()` - Collection discovery
- `select_collection()`, `select_engine()` - Interactive selection with validation
- `get_vector_db_client()`, `require_vector_db_client()` - Vector DB access
- `require_typed_config()` - Typed config with error handling
- `info_line()` - Single-line status display for commands

#### Config Loader (`fitz_sage/config/loader.py`)
- `load_engine_config()` - Loads merged config (package defaults + user overrides)
- `get_config_source()` - Returns config source for debugging
- Package defaults in `fitz_sage/engines/<engine>/config/default.yaml`

#### Collection Existence Warnings (`fitz_sage/cli/commands/query.py`)
- `_warn_if_collection_missing()` - Checks collection before query
- Warns when no collections exist: "Run 'fitz ingest ./docs' first"
- Warns when specified collection not found with available alternatives
- Warns when collection is empty (0 documents)

#### Engine Command (`fitz_sage/cli/commands/engine.py`)
- `fitz engine` - View or set default engine
- `fitz engine --list` - List all available engines
- Interactive selection with card-based UI
- Persists default engine to `.fitz/config.yaml`

#### Instrumentation System (`fitz_sage/core/instrumentation.py`)
- `BenchmarkHook` protocol for plugin performance measurement
- `register_hook()` / `unregister_hook()` - Thread-safe hook management
- `instrument()` decorator for method-level timing
- `create_instrumented_proxy()` - Transparent proxy wrapper for plugins
- Zero overhead when no hooks registered
- Tracks: layer, plugin name, method, duration, errors

#### Enterprise Plugin Discovery (`fitz_sage/cli/cli.py`)
- Auto-discovers `fitz-sage-enterprise` package when installed
- Adds `fitz benchmark` command from enterprise module
- Clean separation: core features in `fitz-sage`, advanced features in enterprise

#### CLI Map Tool (`tools/cli_map/`)
- New tool for analyzing CLI command structure
- Generates visual maps of command hierarchy

### 🔄 Changed

- **Engine rename**: `classic_rag` → `fitz_rag` for clearer branding
- **Quickstart simplified**: Removed `--engine` flag, focuses on fitz_rag for true zero-friction
- **README updated**: Documents auto-detection cascade and first-time experience
- **CLI commands**: All commands now use CLIContext instead of direct config loading
- **Documentation consolidated**: Removed outdated docs (CLARA.md, MIGRATION.md, release notes)

### 🐛 Fixed

- Quickstart no longer prompts for provider when API key is in environment
- Query command now warns about missing collections instead of returning "I don't know"
- Windows API key saving works correctly (uses `.fitz/.env` instead of shell config)

---

## [0.4.4] - 2025-12-30

### 🎉 Highlights

**GraphRAG Engine** - Full implementation of Microsoft's GraphRAG paradigm. Extract entities and relationships, build knowledge graphs, detect communities, and use local/global/hybrid search for relationship-aware retrieval.

**CLaRa Engine Rework** - Major refactoring of the compressed RAG engine with improved architecture and configuration.

**CLI Modernization** - Complete restructure of CLI UI into modular components for better maintainability and user experience.

**Semantic Constraints** - Constraint plugins now use embedding-based semantic matching instead of regex patterns, enabling language-agnostic conflict and causality detection.

### 🚀 Added

#### GraphRAG Engine (`fitz_sage/engines/graphrag/`)
- `GraphRAGEngine` - Knowledge graph-based retrieval engine
- Entity and relationship extraction via LLM (`graph/extraction.py`)
- Knowledge graph storage with NetworkX backend (`graph/storage.py`)
- Community detection using Louvain algorithm (`graph/community.py`)
- Community summarization for high-level insights
- Local search - find specific entities and relationships (`search/local.py`)
- Global search - query across community summaries (`search/global_search.py`)
- Hybrid search - combine local and global approaches
- Persistent storage via JSON serialization
- `fitz_sage/engines/graphrag/config/schema.py` - Full configuration schema

#### Semantic Matching (`fitz_sage/core/guardrails/semantic.py`)
- `SemanticMatcher` class for embedding-based concept detection
- Language-agnostic causal query detection
- Semantic conflict detection across chunks
- Configurable similarity thresholds
- Works with any embedding provider

#### CLI UI Modules (`fitz_sage/cli/ui/`)
- `console.py` - Shared Rich console instance
- `display.py` - Answer and result display formatting
- `engine_selection.py` - Interactive engine selection UI
- `output.py` - Structured output formatting
- `progress.py` - Progress bars and status indicators
- `prompts.py` - User input prompts and confirmations

#### Other
- `fitz_sage/cli/utils.py` - Shared CLI utilities
- `examples/clara_demo.py` - CLaRa engine demonstration
- `tests/test_graphrag_engine.py` - Comprehensive GraphRAG tests

### 🔄 Changed

- **CLaRa engine**: Major refactoring of `fitz_sage/engines/clara/engine.py` with improved architecture
- **CLI commands**: Enhanced `chat`, `ingest`, `init`, `query`, `quickstart` with new UI modules
- **Constraint plugins**: Refactored to use `SemanticMatcher` instead of regex patterns
  - `CausalAttributionConstraint` - Now uses semantic causal evidence detection
  - `ConflictAwareConstraint` - Now uses semantic conflict detection
  - `InsufficientEvidenceConstraint` - Simplified implementation
- **Hierarchy enricher**: Now accepts optional `SemanticMatcher` for conflict detection
- **Config loaders**: Improved engine configuration loading

### 🐛 Fixed

- Contract map tool no longer shows `<unknown>` SyntaxWarnings (added filename to ast.parse)
- Excluded `clara_model_cache` from contract map scanning
- Qdrant tests updated for YAML-based plugin system

---

## [0.4.3] - 2025-12-29

### 🎉 Highlights

**REST API** - New `fitz serve` command launches a FastAPI server with endpoints for query, ingest, and collection management. Build integrations without touching Python.

**SDK Module** - New `fitz_sage.sdk` provides a simplified high-level API for programmatic use. Import `from fitz_sage import Fitz` for quick access.

**Package Rename** - `fitz_sage/ingest/` renamed to `fitz_sage/ingestion/` for clearer naming. Adds new `reader` module for document reading abstraction.

### 🚀 Added

#### REST API (`fitz_sage/api/`)
- `fitz serve` - Launch FastAPI server for HTTP access
- `POST /query` - Query the knowledge base
- `POST /ingest` - Ingest documents
- `GET /collections` - List collections
- `GET /health` - Health check endpoint
- Dependency injection via `fitz_sage/api/dependencies.py`
- Pydantic schemas in `fitz_sage/api/models/schemas.py`

#### SDK Module (`fitz_sage/sdk/`)
- `Fitz` class as unified entry point
- Re-exported from `fitz_sage` package root
- Simplified API for common operations

#### Reader Module (`fitz_sage/ingestion/reader/`)
- `ReaderEngine` for document loading
- Plugin-based reader system
- `local_fs` plugin for local file reading

### 🔄 Changed

- **Package rename**: `fitz_sage/ingest/` → `fitz_sage/ingestion/`
- **Chunk model**: Moved from `fitz_sage/engines/fitz_rag/models/chunk.py` to `fitz_sage/core/chunk.py` for shared use across engines
- **Core exports**: `Chunk` now exported from `fitz_sage.core`

---

## [0.4.2] - 2025-12-28

### 🎉 Highlights

**Knowledge Map** - New `fitz map` command generates an interactive HTML visualization of your knowledge base. View document clusters, explore relationships, and identify coverage gaps. [EXPERIMENTAL]

**Hierarchical RAG** - New enrichment mode that generates multi-level summaries from your content. Groups related chunks and creates hierarchical context for improved retrieval.

**Fast/Smart Model Tiers** - LLM plugins now support two model tiers: "smart" for user-facing queries (best quality) and "fast" for background tasks like enrichment (best speed).

### 🚀 Added

#### Knowledge Map Visualization (`fitz_sage/map/`)
- `fitz map` - Generates interactive HTML knowledge graph
- Automatic clustering of related content
- Gap detection to identify missing coverage
- 2D projection of embeddings for visualization
- State caching for faster regeneration
- `--similarity-threshold` to control edge density
- `--rebuild` to force fresh embedding fetch
- `--no-open` to skip browser launch

#### Hierarchical Enrichment (`fitz_sage/ingest/enrichment/hierarchy/`)
- **HierarchyEnricher**: Generates multi-level summaries from chunks
- **ChunkGrouper**: Groups chunks by source file or custom rules
- **ChunkMatcher**: Filters chunks by path patterns
- Simple mode (zero-config) with smart defaults
- Rules mode for power-users with custom configuration
- Centralized prompts in `fitz_sage/prompts/hierarchy/`

#### Content Type Detection (`fitz_sage/ingest/detection.py`)
- Auto-detects codebase vs document corpus
- Recognizes project markers (pyproject.toml, package.json, Cargo.toml, etc.)
- Selects appropriate enrichment strategy automatically

#### LLM Model Tiers
- `models.smart` and `models.fast` in YAML plugin defaults
- `tier="smart"` or `tier="fast"` parameter for client creation
- Smart defaults: `command-a-03-2025` (Cohere), `gpt-4o` (OpenAI)
- Fast defaults: `command-r7b-12-2024` (Cohere), `gpt-4o-mini` (OpenAI)

#### Comprehensive CLI Tests
- `test_cli_chat.py` - Chat command tests
- `test_cli_collections.py` - Collection management tests
- `test_cli_config.py` - Config command tests
- `test_cli_doctor.py` - System diagnostics tests
- `test_cli_ingest.py` - Ingestion pipeline tests
- `test_cli_init.py` - Initialization tests
- `test_cli_map.py` - Knowledge map tests
- `test_cli_query.py` - Query command tests
- `test_local_llm_*.py` - Local LLM runtime tests

### 🔄 Changed

- Chunker plugins reorganized: `simple.py` and `recursive.py` moved to `plugins/default/`
- `fitz ingest` now supports `-H/--hierarchy` flag for hierarchical enrichment
- Contract map tool refactored with improved autodiscovery
- YAML plugin `defaults.model` replaced with `defaults.models.{smart,fast}` structure

### 🐛 Fixed

- Various fixes to contract map analysis
- Improved chunking router registry handling

---

## [0.4.1] - 2025-12-27

### 🐛 Fixed

- Minor fixes and improvements

---

## [0.4.0] - 2025-12-26

### 🎉 Highlights

**Conversational RAG** - New `fitz chat` command for interactive multi-turn conversations with your knowledge base. Each turn retrieves fresh context while maintaining conversation history.

**Enrichment Pipeline** - New semantic enrichment system that enhances chunks with LLM-generated summaries and produces project-level artifacts for improved retrieval context.

**Batch Embedding** - Automatic batch size adjustment with recursive halving on failure. Significantly faster ingestion for large document sets.

**Collection Management CLI** - New `fitz collections` command for interactive vector DB management.

### 🚀 Added

#### Enrichment System (`fitz_sage/ingest/enrichment/`)
- **EnrichmentPipeline**: Unified entry point for all enrichment operations
- **ChunkSummarizer**: LLM-generated descriptions for each chunk to improve search
- **Artifact Generation**: Project-level insights stored and retrieved with queries
  - `architecture_narrative` - High-level codebase description
  - `data_model_reference` - Data structures and models
  - `dependency_summary` - External dependency overview
  - `interface_catalog` - Public APIs and interfaces
  - `navigation_index` - Codebase navigation guide
- **Context Plugins**: File-type specific context builders (Python, generic)
- **SummaryCache**: Hash-based caching to avoid re-summarizing unchanged content
- **EnrichmentRouter**: Routes documents to appropriate enrichers by file type

#### Batch Embedding
- `embed_batch()` method on `EmbeddingClient`
- Automatic batch size adjustment (starts at 96)
- Recursive halving on API failures
- Progress logging per batch

#### Conversational Interface
- `fitz chat` - Interactive conversation with your knowledge base
- `-c, --collection` option to specify collection directly
- Collection selection on startup (prompts if not specified)
- Per-turn retrieval with conversation history (last 15 messages)
- Rich UI with styled panels for user/assistant messages
- `display_sources()` utility for consistent source table display (vector score, rerank score, excerpt)
- Graceful exit handling (Ctrl+C, 'exit', 'quit')

#### Documentation
- Expanded CLI documentation in `docs/CLI.md` with chat command examples

#### CLI Improvements
- `fitz collections` - Interactive collection management
- Enhanced `fitz_sage/cli/ui.py` with Rich console utilities
- Improved ingest command with enrichment support

#### Retrieval Pipeline
- `ArtifactFetchStep` - Prepends artifacts to every query result (score=1.0)
- Artifacts provide consistent codebase context for all queries

### 🔄 Changed

- Ingest executor now integrates enrichment pipeline
- Ingestion state schema includes enrichment metadata
- README simplified and updated

---

## [0.3.6] - 2025-12-23

### 🎉 Highlights

**Quickstart Command** - Zero-friction entry point for new users. Get a working RAG system in ~5 minutes with just `pip install fitz-sage` and `fitz quickstart`.

**Incremental Ingestion** - Content-hash-based incremental ingestion that skips unchanged files. State-file-authoritative architecture enables user-implemented vector DB plugins without requiring scroll/filter APIs.

**File-Type Based Chunking** - Intelligent routing to specialized chunkers based on file extension. Markdown, Python, and PDF each get purpose-built chunking strategies.

**Epistemic Safety Layer** - Constraint plugins and answer modes prevent overconfident answers when evidence is insufficient, disputed, or lacks causal attribution.

**YAML Retrieval Pipelines** - Retrieval strategies now defined in YAML. Compose steps like `vector_search → rerank → threshold → limit` declaratively.

### 🚀 Added

#### Quickstart Experience
- `fitz quickstart` command for zero-config RAG setup
- Interactive mode with path/question prompts
- Direct mode: `fitz quickstart ./docs "question"`
- Auto-prompts for Cohere API key, offers to save to shell config
- Auto-generates `.fitz/config.yaml` on first run
- Uses Cohere + local FAISS (no external services required)

#### Incremental Ingestion System
- Content-hash-based file tracking in `.fitz/ingest.json`
- Files skipped if content hash matches previous ingestion
- `--force` flag to bypass skip logic and re-ingest everything
- `FileScanner`: Walks directories, filters by supported extensions
- `Differ`: Computes ingestion plan (new/changed/deleted files)
- `DiffIngestExecutor`: Orchestrates parse → chunk → embed → upsert
- `IngestStateManager`: Persists and queries ingestion state

#### File-Type Based Chunking
- `ChunkingRouter`: Routes documents to file-type specific chunkers
- Per-extension chunker configuration via `by_extension` map
- Config ID tracking (`chunker_id`, `parser_id`, `embedding_id`) for re-chunking detection
- `MarkdownChunker`: Splits on headers, preserves code blocks
- `PythonCodeChunker`: AST-based splitting by class/function, includes imports
- `PdfSectionChunker`: Detects ALL CAPS headers, numbered sections, keyword sections

#### Constraint Plugin System
- `ConflictAwareConstraint`: Detects contradicting classifications across chunks
- `InsufficientEvidenceConstraint`: Blocks confident answers when evidence is weak
- `CausalAttributionConstraint`: Prevents implicit causality synthesis
- `ConstraintResult` with `allow_decisive_answer`, `reason`, `signal` fields

#### Answer Mode System
- `AnswerMode` enum: `CONFIDENT`, `QUALIFIED`, `DISPUTED`, `ABSTAIN`
- `AnswerModeResolver`: Maps constraint signals to answer mode
- Mode-specific LLM instruction prefixes for epistemic tone control
- `mode` field added to `RGSAnswer` and core `Answer`

#### YAML Retrieval Pipelines
- `dense.yaml` and `dense_rerank.yaml` pipeline definitions
- Modular retrieval steps: `vector_search`, `rerank`, `threshold`, `limit`, `dedupe`
- `RetrievalPipelineFromYaml` with `retrieve()` method
- Step registry with `get_step_class()` and `list_available_steps()`

#### CLI Improvements
- `fitz init` prompts for chunking configuration
- `fitz ingest` loads chunking config from `fitz.yaml`
- `fitz query --retrieval/-r` flag for retrieval strategy selection
- Shared `display_answer()` for consistent output formatting

### 🔄 Changed

- Config field `retriever` → `retrieval` across codebase
- State schema requires `chunker_id`, `parser_id`, `embedding_id` fields
- `IngestStateManager.mark_active()` requires config ID parameters
- `DiffIngestExecutor` takes `chunking_router` instead of single chunker
- FAISS moved to base dependencies (not optional)

### 🗑️ Deprecated

- `OverlapChunker`: Use `SimpleChunker` with `chunk_overlap` instead

### 🐛 Fixed

- Threshold regression for temporal/causal queries (reordered pipeline steps)
- Plugin discovery paths for YAML-based plugins
- Windows path separator issue in scanner tests
- Contract map now correctly discovers all 25 plugins

---

## [0.3.5] - 2025-12-21

### 🎉 Highlights
**Plugin Schema Standardization** - All LLM plugin YAMLs now follow an identical structure with master schema files as the single source of truth. Adding new providers is now more predictable and self-documenting.

**Generic HTTP Vector DB Plugin System** - HTTP-based vector databases (Qdrant, Pinecone, Weaviate, Milvus) now work with just a YAML config drop - no Python code needed. The same plugin interface works for both HTTP and local vector DBs.

### 🚀 Added
- **Master schema files** for plugin validation and defaults
  - `fitz_sage/llm/schemas/chat_plugin_schema.yaml`
  - `fitz_sage/llm/schemas/embedding_plugin_schema.yaml`
  - `fitz_sage/llm/schemas/rerank_plugin_schema.yaml`
  - `fitz_sage/vector_db/schemas/vector_db_plugin_schema.yaml` - documents all YAML fields for vector DB plugins
- **Schema defaults loader** `fitz_sage/llm/schema_defaults.py` - reads defaults from YAML schemas instead of hardcoding in Python
- **FAISS admin methods** - `list_collections()`, `get_collection_stats()`, `scroll()` for feature parity with HTTP-based vector DBs
- **Azure OpenAI embedding plugin** `fitz_sage/llm/embedding/azure_openai.yaml`
- **New vector DB plugins** (YAML-only, no Python needed):
  - `fitz_sage/vector_db/plugins/pinecone.yaml` - Pinecone cloud vector DB
  - `fitz_sage/vector_db/plugins/weaviate.yaml` - Weaviate vector DB
  - `fitz_sage/vector_db/plugins/milvus.yaml` - Milvus vector DB
- **Vector DB base class for local plugins** `fitz_sage/vector_db/base_local.py` - reduces boilerplate when implementing local vector DBs
- **Comprehensive plugin tests** `tests/test_plugin_system.py` covering chat, embedding, rerank, and FAISS
- **Vector DB plugin tests** `tests/test_generic_vector_db_plugin.py` - validates YAML loading, HTTP operations, point transformation, UUID conversion, and auth handling

### 📄 Changed
- **Standardized plugin YAML structure** - All 13 LLM plugins now follow identical section ordering:
```
  IDENTITY → PROVIDER → AUTHENTICATION → REQUIRED_ENV → HEALTH_CHECK → ENDPOINT → DEFAULTS → REQUEST → RESPONSE
```
- **Chat plugins updated**: openai, cohere, anthropic, local_ollama, azure_openai
- **Embedding plugins updated**: openai, cohere, local_ollama, azure_openai
- **Rerank plugins updated**: cohere
- **Renamed** `list_yaml_plugins()` → `list_plugins()` (removed redundant "yaml" prefix)
- **Loader applies defaults** from master schema - missing optional fields get default values automatically
- **Updated `qdrant.yaml`** - added `count` and `create_collection` operations for full feature parity

### 🛠️ Improved
- **Single source of truth** - Field definitions, types, defaults, and allowed values all live in schema YAMLs
- **Self-documenting schemas** - Each field has `description` and `example` in the schema
- **Forward compatibility** - New fields with defaults don't break existing plugin YAMLs
- **Consistent vector DB interface** - FAISS now implements same admin methods as Qdrant, no backend-specific code needed
- **Generic HTTP vector DB loader** - `GenericVectorDBPlugin` executes YAML specs for any HTTP-based vector DB with support for:
  - All standard operations: `search`, `upsert`, `count`, `create_collection`, `delete_collection`, `list_collections`, `get_collection_stats`
  - Auto-collection creation on 404
  - Point format transformation (standard → provider-specific)
  - UUID conversion for DBs that require it (e.g., Qdrant)
  - Flexible auth (bearer, custom headers, optional)
  - Jinja2 templating for endpoints and request bodies
- **`available_vector_db_plugins()`** - lists all available plugins (both YAML and local)

### 🐛 Fixed
- **FAISS missing interface methods** - Added `list_collections()`, `get_collection_stats()`, `scroll()` to match vector DB contract
- **Rerank mock in tests** - Fixed `MockRerankEngine` to return `List[Tuple[int, float]]` instead of flat list

---

## [0.3.4] - 2025-12-19

### 🎉 Pypi-Release

**https://pypi.org/project/fitz-sage/**

---

## [0.3.3] - 2025-12-19

### 🎉 Highlights

**YAML-based Plugin System** - LLM and Vector DB plugins are now defined entirely in YAML, not Python. Adding new providers is now as simple as creating a YAML file.

### 🚀 Added

- **YAML-based LLM plugins**: Chat, Embedding, and Rerank plugins now use YAML specs
  - `fitz_sage/llm/chat/*.yaml` - Chat plugins (OpenAI, Cohere, Anthropic, Azure, Ollama)
  - `fitz_sage/llm/embedding/*.yaml` - Embedding plugins  
  - `fitz_sage/llm/rerank/*.yaml` - Rerank plugins
- **YAML-based Vector DB plugins**: Vector databases now use YAML specs
  - `fitz_sage/vector_db/plugins/qdrant.yaml`
  - `fitz_sage/vector_db/plugins/pinecone.yaml`
  - `fitz_sage/vector_db/plugins/local_faiss.yaml`
- **Generic plugin runtime**: `GenericVectorDBPlugin` and `YAMLPluginBase` execute YAML specs at runtime
- **Provider-agnostic features**: YAML `features` section for provider-specific behavior
  - `requires_uuid_ids`: Auto-convert string IDs to UUIDs
  - `auto_detect`: Service discovery configuration
- **Message transforms**: Pluggable message format transformers for different LLM APIs
  - `openai_chat`, `cohere_chat`, `anthropic_chat`, `ollama_chat`, `gemini_chat`

### 🔄 Changed

- **LLM plugins**: Migrated from Python classes to YAML specifications
- **Vector DB plugins**: Migrated from Python classes to YAML specifications  
- **Plugin discovery**: Now scans `*.yaml` files instead of `*.py` modules
- **fitz_sage/core/registry.py**: Single source of truth for all plugin access

### 🐛 Fixed

- **Qdrant 400 Bad Request**: String IDs now converted to UUIDs automatically
- **Auto-create collection**: Collections created on first upsert (handles 404)
- **Import errors in CLI**: Fixed by adding re-exports to `fitz_sage/core/registry.py`

---

## [0.3.2] - 2025-12-18

### 🔄 Changed

- Renamed config field `llm` → `chat` for clarity (breaking change - regenerate config with `fitz init`)

### 🚀 Added

- `fitz db` command to inspect vector database collections
- `fitz chunk` command to preview chunking strategies
- `fitz query` as top-level command (was `fitz pipeline query`)
- `fitz config` as top-level command (was `fitz pipeline config show`)
- LAN scanning for Qdrant detection in `fitz init`
- Auto-select single provider options in `fitz init`

### 🐛 Fixed

- Contract map now discovers re-exported plugins (local-faiss)
- Contract map health check false positives removed
- Test fixes for `llm` → `chat` rename

---

## [0.3.1] - 2025-01-17

### 🐛 Fixed

- **CLI Import Error**: Fixed misleading error messages when internal fitz modules fail to import
- **Detection Module**: Moved `fitz_sage/cli/detect.py` to `fitz_sage/core/detect.py` as single source of truth for service detection
- **FAISS Detection**: `SystemStatus.faiss` now returns `ServiceStatus` instead of boolean for consistent API
- **Registry Exceptions**: `LLMRegistryError` now inherits from `PluginNotFoundError` for consistent exception handling
- **Invalid Plugin Type**: `get_llm_plugin()` now raises `ValueError` for invalid plugin types (not just unknown plugins)
- **Ingest CLI**: Fixed import of non-existent `available_embedding_plugins` now uses `available_llm_plugins("embedding")`
- **UTF-8 Encoding**: Added encoding declaration to handle emoji characters in error messages on Windows

### 🔄 Changed

- `fitz_sage/core/detect.py` is now the canonical location for all service detection (Ollama, Qdrant, FAISS, API keys)
- `SystemStatus` now has `best_llm`, `best_embedding`, `best_vector_db` helper properties
- CLI modules (`doctor.py`, `init.py`) now import from `fitz_sage.core.detect` instead of `fitz_sage.cli.detect`

---

## [0.3.0] - 2025-12-17

### 🎉 Overview

Fitz v0.3.0 transforms the project from a RAG framework into a **multi-engine knowledge platform**. This release introduces a pluggable engine architecture, the CLaRa engine for compression-native RAG, and a universal runtime for seamless engine switching.

### ✨ Highlights

- **Universal Runtime**: `run(query, engine="clara")` switch engines with one parameter
- **Engine Registry**: Discover, register, and manage knowledge engines
- **Protocol-Based Design**: Implement `answer(Query) -> Answer` to create custom engines
- **CLaRa Engine**: Apple's Continuous Latent Reasoning with 16x-128x document compression

### 🚀 Added

#### Core Contracts (`fitz_sage/core/`)
- `KnowledgeEngine` protocol for paradigm-agnostic engine interface
- `Query` dataclass for standardized query representation with constraints
- `Answer` dataclass for standardized response with provenance
- `Provenance` dataclass for source attribution
- `Constraints` dataclass for query-time limits (max_sources, filters)
- Exception hierarchy: `QueryError`, `KnowledgeError`, `GenerationError`, `ConfigurationError`

#### Universal Runtime (`fitz_sage/runtime/`)
- `run(query, engine="...")` universal entry point
- `EngineRegistry` for global engine discovery and registration
- `create_engine(engine="...")` factory for engine instances
- `list_engines()` to discover available engines
- `list_engines_with_info()` for engines with descriptions

#### CLaRa Engine (`fitz_sage/engines/clara/`)
- `ClaraEngine` full implementation of CLaRa paradigm
- `run_clara()` convenience function for quick queries
- `create_clara_engine()` factory for reusable instances
- `ClaraConfig` comprehensive configuration
- Auto-registration with global engine registry
- 17 passing tests covering all functionality

#### Fitz RAG Engine (`fitz_sage/engines/fitz_rag/`)
- `FitzRagEngine` wrapper implementing `KnowledgeEngine`
- `run_fitz_rag()` convenience function
- `create_fitz_rag_engine()` factory function
- Auto-registration with global engine registry

### 🔄 Changed

#### Public API (BREAKING)
- Entry points: `RAGPipeline.from_config(config).run()` → `run_fitz_rag()`
- Answer format: `RGSAnswer.answer` → `Answer.text`
- Source format: `RGSAnswer.sources` → `Answer.provenance`
- Chunk ID: `source.chunk_id` → `provenance.source_id`
- Text excerpt: `source.text` → `provenance.excerpt`

#### Directory Structure
```
OLD (v0.2.x):
fitz_sage/
├── pipeline/          # RAG-specific
├── retrieval/         # RAG-specific
├── generation/        # RAG-specific
└── core/              # Mixed concerns

NEW (v0.3.0):
fitz_sage/
├── core/              # Paradigm-agnostic contracts
├── engines/
│   ├── fitz_rag/   # Traditional RAG
│   └── clara/         # CLaRa engine
├── runtime/           # Multi-engine orchestration
├── llm/               # Shared LLM service
├── vector_db/         # Shared vector DB service
└── ingest/            # Shared ingestion
```

### 🐛 Fixed

- Resolved all circular import dependencies
- Fixed import path inconsistencies across modules
- Corrected Provenance field usage (score → metadata)
- Fixed engine registration order to prevent import errors
- Proper lazy imports in runtime to avoid circular dependencies

### 📚 Documentation

- Updated README with multi-engine architecture
- Added CLaRa hardware requirements
- Migration guide for v0.2.x → v0.3.0
- Updated all code examples

### 🧪 Testing

- All existing tests updated and passing
- 17 new tests for CLaRa engine (config, engine, runtime, registration)
- Tests use mocked dependencies (no GPU required for testing)
- Integration tests for engine protocol compliance

### ⚠️ Breaking Changes

1. **Import paths changed**: Update all imports (see Migration Guide)
2. **Public API changed**: Use `run_fitz_rag()` or engine-specific functions
3. **Answer format changed**: `Answer.text` and `Answer.provenance`
4. **No backwards compatibility layer**: Clean break for cleaner codebase

### 📦 Dependencies

New optional dependencies:
```toml
[project.optional-dependencies]
clara = ["transformers>=4.35.0", "torch>=2.0.0"]
```

---

## [0.2.0] - 2025-12-16

### 🎉 Overview

Quality-focused release with enhanced observability, local-first development, and production readiness improvements.

### ✨ Highlights

- **Contract Map Tool**: Living architecture documentation with automatic quality tracking
- **Ollama Integration**: Use local LLMs (Llama, Mistral, etc.) with zero API costs
- **FAISS Support**: Local vector database for development and testing
- **Production Readiness**: 100% appropriate error handling, zero architecture violations

### 🚀 Added

#### Quality Tools
- Contract map with Any usage analysis
- Exception pattern detection
- Code quality metrics tracking
- Architecture violation detection

#### Local Runtime
- Ollama backend for chat, embedding, rerank
- FAISS local vector database
- Local development workflow

#### Developer Experience
- Enhanced error messages in API clients
- Improved logging for file operations
- Better type hints throughout
- Comprehensive documentation

### 🔄 Changed

- Error handling with comprehensive logging
- Type safety improved (92% clean)
- API error messages with better context

### 📚 Documentation

- Updated README with v0.2.0 features
- Contract Map tool documentation
- Local development guide

---

## [0.1.0] - 2025-12-14

### 🎉 Overview

Initial release of Fitz RAG framework.

### 🚀 Added

- Core RAG pipeline
- OpenAI, Azure, Cohere LLM plugins
- Qdrant vector database integration
- Document ingestion pipeline
- CLI tools for query and ingestion

---

[Unreleased]: https://github.com/yafitzdev/fitz-sage/compare/v0.13.0...HEAD
[0.13.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.12.0...v0.13.0
[0.12.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.11.0...v0.12.0
[0.11.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.10.4...v0.11.0
[0.10.4]: https://github.com/yafitzdev/fitz-sage/compare/v0.10.3...v0.10.4
[0.10.3]: https://github.com/yafitzdev/fitz-sage/compare/v0.10.2...v0.10.3
[0.10.2]: https://github.com/yafitzdev/fitz-sage/compare/v0.10.1...v0.10.2
[0.10.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.10.0...v0.10.1
[0.10.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.9.0...v0.10.0
[0.9.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.8.1...v0.9.0
[0.8.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.8.0...v0.8.1
[0.8.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.7.0...v0.7.1
[0.7.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.6.2...v0.7.0
[0.6.2]: https://github.com/yafitzdev/fitz-sage/compare/v0.6.1...v0.6.2
[0.6.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.6.0...v0.6.1
[0.6.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.5.2...v0.6.0
[0.5.2]: https://github.com/yafitzdev/fitz-sage/compare/v0.5.1...v0.5.2
[0.5.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.5.0...v0.5.1
[0.5.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.5...v0.5.0
[0.4.5]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.4...v0.4.5
[0.4.4]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.3...v0.4.4
[0.4.3]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.2...v0.4.3
[0.4.2]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.6...v0.4.0
[0.3.6]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.5...v0.3.6
[0.3.5]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.4...v0.3.5
[0.3.4]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/yafitzdev/fitz-sage/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yafitzdev/fitz-sage/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz-sage/releases/tag/v0.1.0

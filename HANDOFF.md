# fitz-sage handoff

> Written 2026-05-10 at the end of a long refactor session. The next
> session has concrete cleanup tasks. **Read this whole file before
> touching anything** — there are architectural commitments that look
> "weird" if you don't have the context, and the temptation to "helpfully
> reintroduce" deleted features is real. Don't.

## TL;DR — what fitz-sage is now

**One chat-protocol HTTP endpoint, no embeddings.** Retrieval is BM25 +
KRAG typed-unit routing (code symbols, sections, tables) + LLMReranker.
The governance cascade is the safety net for cases where lower recall
would otherwise produce a confidently-wrong answer.

The full quickstart is:

```bash
# Local — recommended
llama-server -m qwen2.5-7b-instruct.gguf --port 8080 &
fitz query "..." --source ./docs

# Cloud (any OpenAI-compatible URL)
fitz query "..." --endpoint https://api.together.xyz/v1 \
                 --model meta-llama-3.1-70b \
                 --api-key-env TOGETHER_API_KEY
```

## Architectural rules — DO NOT VIOLATE

These are decisions the user made explicitly. They are not "preferences"
or "current state, may change later." They are the architecture.

1. **No embeddings, anywhere.** Don't reintroduce `OpenAICompatEmbedding`,
   don't add a "hybrid mode opt-in," don't suggest "let's keep embeddings
   as an optional accelerator." The user's exact words: *"i found
   embeddings to be clunky in almost all real life production scenarios.
   the retrieval intelligence stack is doing the heavy lifting anyways.
   i only had the embedding because i always thought embeddings and rag
   belong together. but they dont"*

2. **One chat-protocol implementation.** OpenAI HTTP only. No Cohere
   chat, no Anthropic chat, no Ollama-specific provider. Provider
   names like `openai` / `azure_openai` are *URL+API-key presets* over
   the canonical `endpoint` provider, not separate code paths. The
   removed names (`ollama`, `cohere`, `anthropic`) raise `ValueError`
   with migration text.

3. **No legacy shims.** When you remove something, remove it. Don't
   leave a deprecation wrapper "for backward compatibility" unless the
   user asks.

4. **Don't run fitz-gov to "validate chat-only mode."** It bypasses
   retrieval by design (line 251 of `fitz_sage/evaluation/benchmarks/fitz_gov.py`:
   `# Run with injected contexts (bypass retrieval for controlled testing)`).
   It tests the governance cascade only. The same scores come out
   whether you have embeddings or not. This is a wrong-instrument
   trap; previous Claude sessions already fell into it.

5. **Don't burn money on cloud APIs without authorization.** OpenAI
   key in env may be invalid (it was, in the smoke test). The user has
   LM Studio at `http://localhost:1234/v1` with `qwen3.6-27b` (chat)
   and `text-embedding-nomic-embed-text-v1.5` (embed — not used by
   fitz-sage anymore but loaded). 32GB VRAM (RTX 5090). Don't spend
   on cloud unless the user said so.

## Recent commits (the trail)

```
b936b45d  refactor(retrieval): commit to single-mode architecture, no embeddings anywhere
39e6ac02  fix(engine): guard three remaining self._embedder.embed_batch sites for chat_only
23fb8606  feat(retrieval): chat_only Phase B — ingestion path skips embedding generation
24c2b078  feat(retrieval): chat_only mode (Phase A) — query-time embedding-free retrieval
fb64bacf  docs(readme): rewrite quickstart for the OpenAI-compatible endpoint architecture
0cf3a6e6  feat(rerank): LLMReranker backend; default rerank: 'llm'
969bfb43  feat(cli): --endpoint / --model / --api-key-env flags + endpoint-aware firstrun
59e9bd89  refactor(llm): single OpenAI-compat protocol; remove ollama/cohere/anthropic providers
```

`b936b45d` is the most recent and the most aggressive. Read its commit
message in full before working on follow-ups (`git show b936b45d --stat
&& git log -1 b936b45d --format=%B`).

## Current state

- **Production code:** consistent. End-to-end smoke test
  (`.smoke_test/smoke_test.py`) ran successfully against LM Studio
  before this commit. chat-only ingest is ~10× faster than the old
  hybrid mode (no embedding generation). See `.smoke_test/results.json`
  for the actual content test results.
- **Unit tests:** **2010 pass, 38 fail, 4 skipped** as of `b936b45d`.
  All 38 failures are tests of code paths that no longer exist
  (semantic merge, HyDE, cloud cache embedding flow, three-way merge
  with semantic leg, engine init asserting `get_embedder` was called).
  None are real regressions.
- **Integrations tests** (`tests/unit/integrations/`) are skipped
  because of a pre-existing `langchain-core` import issue, not our work.
- **Smoke test:** `.smoke_test/smoke_test.py` is committed and runs
  the full pipeline against `tests/e2e_krag/fixtures_rag/` with five
  sanity queries. Use `lms ls` and `lms ps` to check what's loaded
  in LM Studio; load with `lms load <model> --identifier smoke-chat`.
  Verify: `curl http://localhost:1234/v1/models`.

## Follow-up tasks (priority order)

The next session should work top-down through this list. Each task is
self-contained — you don't need to ask the user before starting any of
them. The decisions are made; this is execution.

---

### Task 1 — Make the test suite green

**Goal:** 2048/2048 unit tests pass (excluding the pre-existing
`integrations` skip).

The 38 failing tests are listed at the end of this section. They fall
into four buckets. Triage rule: **if a test exclusively exercises a
deleted feature, delete it. Otherwise rewrite it to test current
behavior.**

#### Bucket A — delete entirely (feature is gone)

These tests test code that no longer exists. Delete the test file or
the affected test classes.

- `tests/unit/test_section_search.py::TestHybridMerge` — tests the
  `_merge_results(bm25, semantic, ...)` signature of
  `SectionSearchStrategy._merge_results`. That method was renamed to
  `_score_results(bm25)` (single leg). Delete `TestHybridMerge` class.
- `tests/unit/test_krag_table_search.py::TestTableSearchStrategy::test_retrieve_semantic_match`
  and `test_retrieve_hybrid_merge` — semantic search is gone.
  TableSearchStrategy is keyword-only. Delete both tests.
- `tests/unit/test_code_search.py::TestCodeSearchStrategy::test_retrieve_combines_keyword_and_semantic`
  — same. Replace with a `test_retrieve_combines_keyword_and_bm25`
  test that drops the `search_by_vector` mock.

#### Bucket B — update assertions (test still meaningful)

These tests have a meaningful behavioral question, but their assertions
reference removed fields (`embedding`, `embedding_base_url`,
`fallback_to_chunks`, `enable_hyde`).

- `tests/unit/test_config_loader.py::test_load_config_from_defaults`
  and `test_config_none_for_disabled` — drop assertions on removed
  fields. Read `fitz_sage/engines/fitz_krag/config/schema.py` to see
  what fields actually exist now.
- `tests/unit/test_default_config_loads_and_validates_minimally.py::test_default_config_loads_and_validates_base_schema`
  — same.
- `tests/unit/test_krag_registration.py::TestKragRegistration::test_config_loader_loads_defaults`
  — same.

#### Bucket C — engine init asserts `get_embedder` was called

These mock `get_embedder` and assert it was called once. After the
refactor, `get_embedder` is never called. Either:

1. Remove the `mock_get_embedder` patch + the corresponding assertion, OR
2. Drop the test if its sole purpose was to verify embedder construction.

Affected:

- `tests/unit/test_krag_engine.py::TestEngineInit::test_init_creates_components`
- All of `tests/unit/test_krag_engine.py::TestAnswer::*` — these mock
  the full engine pipeline; they need their `mock_get_embedder` patches
  cleaned up.
- All of `tests/unit/test_krag_detection.py::TestRouterDetection::*`
  and `TestEngineAnswerDetectionFlow::*` and `TestTemporalTagging::*`
  — same pattern.
- `tests/unit/test_krag_query_rewriting.py::TestQueryRewriting::*`
  — same.
- `tests/unit/test_krag_guardrails.py::TestAnswerModePassthrough::*`
  — same.
- `tests/unit/test_retrieval_wiring.py::TestSectionFreshnessBoostWithRecency::test_section_freshness_boost_with_recency`
  — same.
- `tests/unit/test_ingest_executor.py::TestRunDiffIngest::test_runs_ingestion`
  — same.
- `tests/unit/test_ingest_timing.py::test_ingest_timing_detailed`
  and `test_ingest_with_real_embedder_mock` — the second test name is
  literally about the embedder mock, so it can be deleted; the first
  needs the mock_get_embedder patch removed.

The pattern to look for in each test file:

```python
@patch("fitz_sage.llm.client.get_embedder")
@patch("fitz_sage.llm.client.get_chat")
...
def test_something(self, ..., mock_get_embedder, mock_get_chat, ...):
    mock_get_embedder.return_value.dimensions = 1024  # delete this line
    ...
    mock_get_embedder.assert_called_once_with(...)  # delete this assertion
```

In all cases: drop the `@patch("...get_embedder")` decorator, drop the
corresponding parameter from the function signature, drop any
`mock_get_embedder.*` lines.

#### Verification

After each file fix:

```bash
python -m pytest tests/unit/<the_file>.py -q
```

When you think the whole bucket is done:

```bash
python -m pytest tests/unit -q --ignore=tests/unit/integrations
```

Goal output: `2048 passed, 4 skipped`.

---

### Task 2 — Delete `OpenAICompatEmbedding` and the embedding routing

**Goal:** Remove the LLM-layer embedding surface that no longer has
production callers.

#### Files to edit

- `fitz_sage/llm/providers/openai_compat.py` — delete the
  `OpenAICompatEmbedding` class and `OPENAI_EMBEDDING_MODEL` constant.
  Remove from `__all__`.
- `fitz_sage/llm/providers/__init__.py` — remove
  `OpenAICompatEmbedding` from the import + `__all__`.
- `fitz_sage/llm/client.py` — delete `get_embedder` function.
- `fitz_sage/llm/__init__.py` — remove `get_embedder` from re-exports.
- `fitz_sage/llm/config.py` — delete `create_embedding_provider`
  function entirely. Drop the `EmbeddingProvider` import.
- `fitz_sage/llm/providers/base.py` — delete the `EmbeddingProvider`
  protocol.

#### Tests to update

- `tests/unit/llm/test_endpoint_provider.py` — delete
  `TestEndpointEmbeddingProvider` class entirely.
- `tests/unit/llm/test_client.py` — delete the embedder tests in
  `TestEndpointProvider` (e.g. `test_endpoint_embedder_local`).
- `tests/unit/llm/test_config.py` — delete embedding-related test
  cases that reference `create_embedding_provider`.

#### Verification

```bash
python -m pytest tests/unit/llm -q
```

Then full suite. Should still be all green.

#### Ripple effects to check

- `tests/integration/cloud_fixtures.py` may reference `get_embedder`.
- `fitz_sage/cli/services/init_service.py::get_default_model` — has a
  branch on `plugin_type == "embedding"`. Delete the branch.

---

### Task 3 — Drop pgvector + `fitz_sage/vector_db/` subpackage

**Goal:** Remove the vector-DB plugin abstraction and the pgvector
extension dependency. Vector columns and HNSW indexes go from the schema.

#### Files to delete

- `fitz_sage/vector_db/` — the entire directory (~1.6k LOC). Delete with:
  ```bash
  git rm -r fitz_sage/vector_db/
  ```

#### Schema changes

- `fitz_sage/engines/fitz_krag/ingestion/schema.py`:
  - Drop `summary_vector` columns from `_symbol_index_ddl`,
    `_section_index_ddl`, `_table_index_ddl`.
  - Drop the HNSW DDL functions (`_symbol_hnsw_index_ddl`,
    `_section_hnsw_index_ddl`, `_table_hnsw_index_ddl`).
  - Drop `_validate_vector_dimensions`.
  - Change `ensure_schema(connection_manager, collection, embedding_dim)`
    to `ensure_schema(connection_manager, collection)`. Remove the
    `embedding_dim` parameter throughout.
- `fitz_sage/engines/fitz_krag/engine.py`:
  - Delete `_LEGACY_VECTOR_DIM = 1024` constant.
  - Remove the dim arg from the `ensure_schema(...)` call.
- `fitz_sage/engines/fitz_krag/ingestion/pipeline.py`:
  - Remove the `_LEGACY_VECTOR_DIM` import + dim arg.
- Stores (`symbol_store.py`, `section_store.py`, `table_store.py`):
  - Drop the `search_by_vector` methods.
  - Drop the `update_vector` and `update_vectors_by_file` methods (or
    keep them as no-ops if any caller still references them — grep
    first).
  - Drop vector columns from the SELECT/INSERT statements.

#### Config schema

- `fitz_sage/engines/fitz_krag/config/schema.py`:
  - Delete `vector_db` field.
  - Delete `vector_db_kwargs` field (and the `PluginKwargs` type if no
    other field uses it).
- `fitz_sage/engines/fitz_krag/config/default.yaml`:
  - Delete `vector_db: pgvector` and `vector_db_kwargs:` block.

#### pyproject.toml

- Drop `pgvector` from dependencies (search for it; should be in
  `[project.dependencies]`).
- `pgserver` stays (PostgreSQL is still used for FTS).

#### Worker cleanup

`fitz_sage/engines/fitz_krag/progressive/worker.py` — the four
`_embed_*` methods are currently no-ops. Now you can delete them and
delete the `EMBEDDED` state from the progressive state machine.

- Find the `FileState` enum (probably in `progressive/manifest.py`).
  Drop `EMBEDDED`.
- Find every reference to `FileState.EMBEDDED` and either delete the
  surrounding logic or transition directly from `SUMMARIZED` to
  whatever comes next (probably "complete").

#### Tests

- `tests/unit/vector_db/` — delete the entire directory.
- Any test file referencing `pgvector`, `search_by_vector`, or
  `summary_vector` in mocks needs updating.

#### Verification

```bash
# Suite still green
python -m pytest tests/unit -q --ignore=tests/unit/integrations

# Smoke test still works
# (Make sure llama-server is running on 8080 OR LM Studio is loaded;
# update .smoke_test/smoke_test.py LM_STUDIO_URL if needed)
python .smoke_test/smoke_test.py
```

#### Migration note for the user's existing collections

If they have any collections previously ingested with embeddings,
those will still work (vector columns are NULL but present). After
this task, the vector columns are gone. **Don't auto-migrate.** Add
a CLI command `fitz collections migrate-no-vectors <name>` or similar
that drops the vector columns from existing collections, and document
it as a one-time step. Or just tell users to delete and re-ingest.

---

### Task 4 — Re-key cloud cache on text hash, OR delete cloud cache

**Goal:** `_check_cloud_cache` and `_store_cloud_cache` in
`fitz_sage/engines/fitz_krag/engine.py` currently return None
(disabled). The cloud cache used query embeddings as the
similarity-keyed cache key. Without embeddings, there are two paths:

#### Option A — re-key on text hash (keep the feature)

- `fitz_sage/cloud/cache_key.py`: add a `compute_query_hash(query: str)
  -> str` function (just SHA-256 of normalized query text).
- `fitz_sage/cloud/client.py::CloudClient.lookup_cache` and
  `store_cache`: change the signature to take `query_hash: str`
  instead of `query_embedding: list[float]`.
- `engine.py`: re-enable the methods using query_hash.

#### Option B — delete cloud cache entirely

- Delete `fitz_sage/cloud/` subpackage.
- Delete `cloud` field from `FitzKragConfig`.
- Delete the `_cloud_client`, `_check_cloud_cache`, `_store_cloud_cache`
  on the engine.

The user hasn't expressed a preference. Default to **Option B** unless
they object — the cloud feature is one of the "ambitious surface
area" items the user has been pruning. If you go with B, mention it
in the commit message so they can flag if they wanted A.

---

### Task 5 — `firstrun.py` cleanup

`fitz_sage/core/firstrun.py` still classifies discovered models into
chat vs embedding via `_EMBEDDING_PATTERNS`. With embeddings gone, we
no longer need to pick an embedding model.

- Remove `embedding_models` list from `DetectedEndpoint`.
- Remove `_classify` (or simplify to just return the id as a chat model).
- Remove `_pick_embedding_model`.
- Update `_configure_from_endpoint` to only write a chat config.
- Update `_configure_from_openai_key` to only set `chat_smart` /
  `chat_balanced` / `chat_fast` (no embedding spec).
- Update the existing tests in `tests/unit/test_firstrun.py` to drop
  embedding model assertions.

---

### Task 6 — Documentation refresh

Update these files to match the no-embeddings architecture:

- `README.md` — already mostly updated in commit `fb64bacf`, but
  search for any remaining `embedding` references and remove them.
  Update the architecture diagram.
- `docs/features/platform/openai-compatible-endpoint.md` — already
  mentions "chat-only mode" but the term is now outdated (it's just
  "the mode"). Rewrite to drop the bimodal framing.
- `docs/features/platform/krag.md` — likely mentions hybrid retrieval.
  Update to reflect BM25 + LLMReranker.
- `docs/CONFIG.md` — drop embedding fields documentation.
- `docs/ARCHITECTURE.md` — diagram + data flow update.

---

## How to verify the whole thing works

After each major task, run:

```bash
# 1. Unit suite green
python -m pytest tests/unit -q --ignore=tests/unit/integrations

# 2. End-to-end smoke test against the user's LM Studio
#    (verify LM Studio is up first: curl http://localhost:1234/v1/models)
python .smoke_test/smoke_test.py

# 3. Schema integrity (after Task 3): inspect that vector columns
#    are gone from a fresh-collection schema
python -c "
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine
import uuid
cfg = FitzKragConfig(collection=f'verify_{uuid.uuid4().hex[:6]}')
e = FitzKragEngine(cfg)
print('Engine init OK')
"
```

## Key context: the user's actual workflow

- **OS:** Windows. Don't suggest `apt-get` or assume `/usr/bin`.
- **Shell:** Git Bash via the Bash tool.
- **VRAM:** RTX 5090, 32 GB. Models loaded via LM Studio CLI (`lms`).
  Currently loaded (as of this handoff): `qwen3.6-27b` as `smoke-chat`,
  `text-embedding-nomic-embed-text-v1.5@q8_0` as `smoke-embed`. The
  embed model is loaded but unused — fine to unload.
- **Other projects:** The user is also working on `expert-prefetch-v3`
  (TEC, llama.cpp MoE inference) and `fitz-forge`. fitz-sage is one of
  several related projects; don't conflate them.
- **Tooling:** `rtk` is the user's token-saving CLI wrapper.
  Project-level CLAUDE.md (read it) requires prefixing commands with
  `rtk` for git/grep/etc. Standard tools (Glob, Grep, Read, Edit) are
  fine without `rtk`.

## Files / paths to know

- `fitz_sage/engines/fitz_krag/engine.py` — the heart. ~1500 LOC.
- `fitz_sage/engines/fitz_krag/config/schema.py` — FitzKragConfig.
- `fitz_sage/engines/fitz_krag/retrieval/strategies/` — three files
  (code, section, table) after `chunk_fallback.py` was deleted.
- `fitz_sage/llm/providers/openai_compat.py` — the only chat-protocol
  implementation.
- `fitz_sage/llm/providers/llm_reranker.py` — LLMReranker.
- `fitz_sage/llm/config.py` — provider routing (presets over the
  single OpenAICompatChat class).
- `fitz_sage/governance/decider.py` — v6 cascade ML classifier.
  Requires `xgboost` and `lightgbm` (already installed in the user's
  env this session).
- `tests/e2e_krag/fixtures_rag/` — small mixed corpus (~9 files) used
  by the smoke test. Includes a known conflict between Finance and HR
  on TechCorp employee count (5,200 vs 4,800), useful for testing
  DISPUTED governance.
- `~/.fitz/fitz_gov_data/data/` — fitz-gov benchmark data already
  downloaded. Categories: abstention (40), dispute (40), grounding (25),
  relevance (25), confidence (30), qualification (40). Total: 200
  cases. **Note:** the README's "2,920 cases" refers to a larger
  dataset that's not what's downloaded here — possibly a synthetic
  augmentation. Don't trust the 2,920 number; trust the file counts.
- `.smoke_test/smoke_test.py` — end-to-end test harness against
  fixtures_rag using LM Studio at port 1234.
- `.smoke_test/results.json` — last run's results (chat_only 3/5
  vs hybrid 2/5 mode-match — but this is now just chat_only since
  hybrid is gone).

## Conversation context for the next session

The previous session arc was:
1. Started: "lets check up on fitz-sage." Status check, then assessment.
2. User decided to drop Ollama-specific provider, single OpenAI-HTTP
   protocol. Done in `59e9bd89`.
3. User asked "is embedding even necessary" — discussed; answer was
   "demote, don't delete" with chat_only mode added in `24c2b078` /
   `23fb8606`.
4. User then said: "isnt that novel?" — discussed novelty honestly
   (combination is unusual, method components aren't new).
5. User said "if we have no embeddings, why do we need pgvector
   then? this is dead aswell" — agreed, but we deferred pgvector
   removal to follow-up.
6. User said "run fitz gov." — discovered fitz-gov bypasses retrieval
   so it can't validate chat_only. Stopped.
7. User said "run fitz-sage on some sanity queries with sanity corpus."
   — built the smoke test, ran via LM Studio, found a real bug
   (`embedder=None` → `'NoneType' object has no attribute embed_batch'`
   in three engine.py sites), fixed in `39e6ac02`.
8. User said "its enough for me to say it works end to end" + the
   architectural-commitment quote about embeddings being clunky.
9. Demolition pass landed as `b936b45d`. Tests cascaded — 38 failures
   tracked here as Task 1.

The last user message was a request for this handoff doc.

## Final thing — don't waste a context window second-guessing

The architecture is committed. The user evaluated and chose. Next
session: execute Task 1, then 2, then 3, then 4 or 5, then 6. Run the
tests after each. Commit each task as its own commit with a clear
message. Don't ask "are you sure about no embeddings?" — they're sure.

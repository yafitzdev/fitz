# fitz-sage handoff

> Updated 2026-05-10 after a deep refactor session that landed Tasks 1,
> 2, and 3 (plus a mid-Task-1 architectural cleanup the original handoff
> didn't anticipate). **Read this whole file before touching anything.**
> The architectural commitments here are decisions, not preferences;
> the temptation to "helpfully reintroduce" deleted features is real.
> Don't.

## TL;DR — what fitz-sage is now

**One chat-protocol HTTP endpoint. No embeddings. No vector DB.**
Retrieval is BM25 + KRAG typed-unit routing (code symbols, sections,
tables) + LLMReranker. Storage is PostgreSQL — `krag_section_index` /
`krag_symbol_index` / `krag_table_index` with `tsvector` GIN indexes,
no `vector` columns, no HNSW. The governance cascade is the safety
net for cases where lower recall would otherwise produce a
confidently-wrong answer.

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

These are decisions the user made explicitly. They are not
"preferences" or "current state, may change later." They are the
architecture.

1. **No embeddings, anywhere.** Don't reintroduce
   `OpenAICompatEmbedding`, don't add a "hybrid mode opt-in," don't
   suggest "let's keep embeddings as an optional accelerator." The
   user's exact words: *"i found embeddings to be clunky in almost
   all real life production scenarios. the retrieval intelligence
   stack is doing the heavy lifting anyways. i only had the embedding
   because i always thought embeddings and rag belong together. but
   they dont"*

2. **One chat-protocol implementation.** OpenAI HTTP only. No Cohere
   chat, no Anthropic chat, no Ollama-specific provider. Provider
   names like `openai` / `azure_openai` are *URL+API-key presets*
   over the canonical `endpoint` provider, not separate code paths.
   The removed names (`ollama`, `cohere`, `anthropic`) raise
   `ValueError` with migration text.

3. **No vector DB.** PostgreSQL is the only storage. `pgvector` is no
   longer a runtime dependency. `fitz_sage/vector_db/` doesn't exist.
   Don't suggest "we could keep pgvector as an option." Vector search
   isn't switched off — it's gone.

4. **No shims, ever.** When you remove a feature, remove the surface
   too. No constructor parameters that nothing reads, no fields
   hardcoded to a constant for caller compat, no exported classes
   with no in-tree caller, no `# legacy field; ignored` comments, no
   `# kept for backward compatibility` shims. Default action when
   removing: delete entirely, then chase down every caller and clean
   them up. **Tests are not the source of truth** — when a test fails
   after a refactor, verify against production code first; if the
   test exercises a deleted feature, delete the test, don't `Mock()`
   your way past a missing argument. The user flagged this twice in
   one session; assume strict enforcement.

5. **Don't run fitz-gov to "validate chat-only mode."** It bypasses
   retrieval by design (line 251 of
   `fitz_sage/evaluation/benchmarks/fitz_gov.py`:
   `# Run with injected contexts (bypass retrieval for controlled testing)`).
   It tests the governance cascade only. The same scores come out
   whether you have embeddings or not. This is a wrong-instrument
   trap; previous Claude sessions already fell into it.

6. **Don't burn money on cloud APIs without authorization.** OpenAI
   key in env may be invalid (it was, in the smoke test). The user
   has LM Studio at `http://localhost:1234/v1` with `qwen3.6-27b`
   loaded as `smoke-chat`. 32GB VRAM (RTX 5090). Don't spend on
   cloud unless the user said so.

## Recent commits (the trail)

```
bdca80ae  refactor(storage): rip out pgvector + the entire vector_db plugin tree
c7dc48a1  refactor(llm): delete the embedding API and every in-tree consumer
42fd4c7d  refactor(config): drop dead semantic/section/table weight fields
65ad6962  refactor: rip out HyDE, ChunkFallbackStrategy, and DiffIngestExecutor surface
3b6b8afc  test(suite): green up the cascade from the no-embeddings refactor
a96b32b1  docs(handoff): write HANDOFF.md for the next session   ← original handoff
b936b45d  refactor(retrieval): commit to single-mode architecture, no embeddings anywhere
```

`bdca80ae` is the most recent. Read these in full before working on
follow-ups — they document the demolition decisions:
`git log --format=%B 3b6b8afc^..HEAD`.

### What landed in this session

- **3b6b8afc** — Task 1 (initial). Greened the 38 failing unit tests
  left by `b936b45d`. Surfaced and fixed two production bugs in
  `retrieval_profile.build_retrieval_profile` (the function was
  reading the now-deleted `config.enable_hyde` and
  `config.fallback_to_chunks`, so `engine.answer()` was crashing on
  every query post-`b936b45d`). This commit had shims in it, called
  out by the user; they were ripped in the next two.
- **65ad6962** — −3,981 / +81. Mid-Task-1 architectural cleanup the
  original handoff didn't list as a separate task: deleted the
  `RetrievalProfile.run_hyde` / `RetrievalProfile.fallback_to_chunks`
  fields entirely, deleted the dead branches in `RetrievalRouter`
  (embedder embed_batch loop, HyDE generation, chunk fallback), the
  dead constructor params (`embedder=`, `hyde_generator=`,
  `chunk_strategy=`), the `query_vector=` and `hyde_vectors=` kwargs
  from every strategy `retrieve()`, the `precomputed_query_vectors`
  retrieve param, the broken `ChunkFallbackStrategy` TYPE_CHECKING
  import. Deleted `fitz_sage/retrieval/hyde/` and
  `fitz_sage/ingestion/diff/` modules entirely. Reverted the `embedder=
  MockEmbedder()` shim adds from the prior commit (the tests they
  shimmed exercised the deleted DiffIngestExecutor — file deletion
  was the right cut).
- **42fd4c7d** — Dropped the dead `semantic_weight` /
  `section_bm25_weight` / `section_semantic_weight` /
  `table_keyword_weight` / `table_semantic_weight` fields from
  `FitzKragConfig`, default.yaml, init template. Kept `keyword_weight`
  and `code_bm25_weight` (those are read by `CodeSearchStrategy`).
  Rewrote `.smoke_test/smoke_test.py` for single-mode (the old file
  built `chat_only` vs `hybrid` passes referencing deleted fields).
- **c7dc48a1** — Task 2. Deleted the LLM-layer embedding API entirely:
  `OpenAICompatEmbedding`, `OPENAI_EMBEDDING_MODEL`,
  `EmbeddingProvider` Protocol, `get_embedder`,
  `create_embedding_provider`. Chased every in-tree consumer to zero
  across 13 files (`cli/context.py`, `cli/commands/*.py`,
  `cli/services/init_service.py`, `core/firstrun.py`,
  `core/detect.py`, `services/fitz_service.py`,
  `retrieval/detection/registry.py`, the krag ingestion TYPE_CHECKING
  imports). Deleted `fitz_sage/retrieval/detection/concept_detector.py`
  and `tools/governance/{extract_features,eval_pipeline}.py`. Note:
  this commit also did most of what the original handoff called
  "Task 5 (firstrun.py cleanup)" — that task is effectively done.
- **bdca80ae** — Task 3. −7,121 / +157. Largest cut. Deleted
  `fitz_sage/vector_db/` (~1.5k LOC), `fitz_sage/backends/local_vector_db/`,
  `fitz_sage/ingestion/state/`. Stripped vector columns + HNSW DDL
  from `ingestion/schema.py`. Stripped `search_by_vector` /
  `update_vector*` / `summary_vector` from all three stores.
  Dropped `_LEGACY_VECTOR_DIM`, `_embed_summaries`,
  `FileState.EMBEDDED` / the `EMBEDDED` state-machine phase, the
  `_embed_*` no-op worker methods. Dropped `vector_db` /
  `vector_db_kwargs` from `FitzKragConfig` + `default.yaml`. Rewrote
  `FitzService.list_collections` / `delete_collection` to use
  `PostgresConnectionManager` directly (psycopg autocommit against
  the postgres admin DB for `DROP DATABASE`). Dropped `pgvector`
  from `pyproject.toml`. Rewrote `evaluation/benchmarks/beir.py`
  to BM25-only.

## Current state

- **Production code:** consistent and verified end-to-end.
- **Unit tests:** **1778 pass, 1 skipped, 0 fail.** The 1-skipped
  module is `tests/unit/integrations/` (pre-existing `langchain-core`
  import issue, not our work). The test count dropped from 2048 over
  the session because dead-feature tests went with the dead features
  — every deletion was deliberate, called out in commit messages.
- **End-to-end smoke** (`.smoke_test/smoke_test.py` against LM
  Studio at `:1234`, `smoke-chat` = `qwen3.6-27b`): engine init clean,
  ingest 0.0s for 9 files, **5/5 queries answered without exceptions**,
  **6/6 expected substrings present**, 3/5 governance mode-match
  (the 2 misses — disputed/abstain — are expected because the smoke
  runs with `enable_guardrails=False` for speed).
- **The architecture:** PostgreSQL-only retrieval. `tsvector` GIN
  indexes on `krag_section_index.content_tsv` (auto-generated from
  title + content) and `krag_symbol_index.content_tsv` (auto-generated
  from name + qualified_name + summary). `search_bm25` uses
  `ts_rank_cd` with parent-context join in
  `SectionSearchStrategy._score_results` (pure RRF, `1 / (60 + rank)`).
  No `vector` columns anywhere. No HNSW. No embedding-model knobs.

## Follow-up tasks (priority order)

**Execution order (confirmed 2026-05-11):** Task 4 → Task 10 →
Task 6 → Task 7 → Task 8 → Task 9. Task 3.5 is **skipped** — Task 10
(postgres → sqlite) subsumes it entirely, so cleaning up the
pgvector surface beforehand is throwaway work.

Each task is self-contained — you don't need to ask the user before
starting any of them. The decisions are made; this is execution.

---

### Task 3.5 — Finish pgvector demolition ⏭ SKIPPED (subsumed by Task 10)

Surfaced 2026-05-11 verifying Task 3 was clean. The pip dep is gone
and `vector_db/` is deleted, but production code still references
pgvector — and `storage/postgres.py:559-564` still runs
`CREATE EXTENSION vector` on init, which is a real runtime bug.

**Skipped by decision (2026-05-11):** Task 10 (postgres → sqlite)
will delete `storage/postgres.py` entirely, along with the
postgres-detection scaffolding in `core/detect.py`, the CLI status
flows, and the vocabulary/entity_graph/sparse stores. Cleaning up
the pgvector surface beforehand is throwaway work.

The edit list below is preserved as a checklist of "things that must
end up deleted, one way or another" — Task 10 should verify nothing
on this list slips through.

#### Edits

- `fitz_sage/storage/postgres.py` — remove the `CREATE EXTENSION
  vector` block (~559-564) and the "pgvector extension initialization"
  comments at lines 9, 226.
- `fitz_sage/core/detect.py` — delete `detect_pgvector()` (197-213),
  the `pgvector: ServiceStatus` field on `SystemEnvironment` (62), the
  `_vector_db_kind()` method (92-93) returning `"pgvector"`, and the
  `pgvector=detect_pgvector()` wiring in the factory (270).
- `fitz_sage/cli/services/init_service.py` — drop `pgvector: Any` (19),
  the `pgvector=system.pgvector` arg (47), the
  `if "pgvector" in plugin_lower:` branch (81-83), and the docstring
  mention (67).
- `fitz_sage/cli/commands/config.py:299-302` — drop the pgvector UI
  status block.
- `fitz_sage/cli/commands/init_wizard.py:207,273` — drop the pgvector
  detection mention and the UI status line.
- `fitz_sage/cli/context.py:438` — drop the "VectorDB: pgvector"
  example from the docstring.
- `fitz_sage/plugin_gen/library_context.py:42` — drop the
  `"pgvector": "pgvector"` mapping.
- `fitz_sage/retrieval/sparse/index.py:8,34,76` — rewrite comments
  ("pgvector chunks table" → "krag indexes").
- `fitz_sage/storage/__init__.py:6` — fix docstring ("PostgreSQL +
  pgvector" → "PostgreSQL with tsvector GIN").

**Leave alone:** `fitz_sage/code/__init__.py:3` and
`fitz_sage/code/retriever.py:38` — those comments correctly describe
what the standalone `code` extra *avoids*.

#### Verification

```bash
python -m pytest tests/unit -q --ignore=tests/unit/integrations
python .smoke_test/smoke_test.py
```

Smoke must still init Postgres cleanly without the extension —
`tsvector` is built into Postgres core; only the `vector` type
needed pgvector.

---

### Task 4 — Delete the cloud feature

Default per the original handoff: **delete entirely** (option B). The
cloud feature is one of the "ambitious surface area" items the user
has been pruning. If you want option A (re-key cloud cache on text
hash) ask first; expect "delete."

#### Files to delete

- `fitz_sage/cloud/` — the entire subpackage.
- `tests/integration/cloud_fixtures.py` — already broken since
  Task 1 (imports the deleted `fitz_sage.ingestion.diff.run_diff_ingest`).
- `tests/integration/test_cloud_cache_e2e.py` — consumer of
  `cloud_fixtures`.
- `tests/integration/conftest.py` — drop cloud-fixture imports.

#### Production updates

- `fitz_sage/engines/fitz_krag/config/schema.py`: drop the `cloud`
  dict field.
- `fitz_sage/engines/fitz_krag/config/default.yaml`: drop any `cloud:`
  section if present.
- `fitz_sage/engines/fitz_krag/engine.py`: drop `self._cloud_client`,
  drop `_check_cloud_cache` and `_store_cloud_cache` (both already
  return None — confirm by grepping `_cloud_client` first), drop the
  call sites in `answer()`.
- `fitz_sage/cli/` — search for any `cloud` flag / command. If a
  `fitz cloud login` or similar exists, delete the command file.

#### Verification

```bash
python -m pytest tests/unit -q --ignore=tests/unit/integrations
python .smoke_test/smoke_test.py
```

Both should pass. The smoke test doesn't exercise cloud, so its
output should be identical to the current baseline.

---

### Task 5 — `firstrun.py` cleanup ✅ DONE

This is already done. `core/firstrun.py` was rewritten in commit
`c7dc48a1` — `DetectedEndpoint` no longer has `embedding_models`,
`_classify` and `_pick_embedding_model` are gone, `write_config`
only emits chat config. Skip this task.

---

### Task 6 — Documentation refresh

**Blocked on Task 10 (sqlite migration).** Don't start this until
Task 10 lands — the docs need to reflect the final SQLite-FTS5
architecture, not the current Postgres-tsvector one.

The code stopped matching the docs three commits ago. Time to fix.

Files to refresh:

- **README.md** — search for `embedding`, `vector_db`, `pgvector`,
  `HyDE`. Update the architecture diagram. The quickstart already
  says "no embeddings" but examples may still mention an embedding
  spec.
- **docs/ARCHITECTURE.md** — diagram + data flow update. The
  retrieval stack is now: query → analyzer → router (BM25
  multi-strategy in parallel) → CrossStrategyRanker → reader →
  context assembler → synthesizer. No embedder, no vector DB, no
  HyDE, no chunk fallback.
- **docs/CONFIG.md** — drop all embedding / vector_db / HyDE fields.
  The current `FitzKragConfig` (post-Task-3) is the source of truth.
- **docs/PLUGINS.md** — drop the embedding plugin / vector DB plugin
  / vector DB schema documentation. The plugin system now has chat,
  vision, rerank, and chunking — nothing else.
- **docs/INGESTION.md** — drop the entire `DiffIngestExecutor`
  walkthrough (lines ~371-400). Replace with the
  `KragIngestPipeline` flow (or just delete the section — the
  engine's `point()` method is the user-facing API now).
- **docs/TROUBLESHOOTING.md** — drop `get_embedder`-based snippets.
- **docs/features/platform/openai-compatible-endpoint.md** —
  embeddings section needs to go; this is now a chat-only
  document.
- **docs/features/platform/krag.md** — likely mentions hybrid
  retrieval. Update to reflect BM25-only.
- **docs/features/retrieval/sparse-search.md** + **hybrid-search.md**
  + **contextual-embeddings.md** — review and either delete or
  rewrite. Hybrid search is gone. Contextual embeddings are gone.
- **CHANGELOG.md** — don't rewrite history. Add a new entry at the
  top documenting the demolition.

---

### Task 7 — Enrichment hierarchy cleanup (new task surfaced by Task 3)

Out of scope for Task 3, but flagged in `bdca80ae`'s commit message.
`fitz_sage/ingestion/enrichment/hierarchy/embedding_provider.py` has
its own `EmbeddingProvider` class (different from the deleted
LLM-layer Protocol). `HierarchyEnricher` takes an `embedder` and
creates an `EmbeddingProvider` if it's truthy. Neither is constructed
by fitz-sage's live krag ingestion path — `KragIngestPipeline` uses
`progressive/worker.py`, which never instantiates the enrichment
pipeline.

Two options:

A. **Delete `fitz_sage/ingestion/enrichment/hierarchy/` and its
   tests entirely.** Same shape as the `DiffIngestExecutor` deletion
   in `65ad6962` — exported but unused.

B. **Repurpose `HierarchyEnricher` for L1/L2 LLM hierarchical
   summaries without embeddings** if the user wants hierarchical
   enrichment back. The current `_generate_hierarchy_symbols` /
   `_generate_hierarchy_sections` in `pipeline.py` already do
   something like this and don't depend on the enrichment module.

Recommend A unless the user objects. Worth asking before swinging.

---

### Task 8 — Integration test cleanup

`tests/integration/` is not collected by the unit suite, but several
files import deleted modules:

- `cloud_fixtures.py:174` — `from fitz_sage.ingestion.diff import run_diff_ingest` (deleted in `65ad6962`).
- `cloud_fixtures.py:177` — `from fitz_sage.llm import get_embedder` (deleted in `c7dc48a1`).
- `test_governance_constraints.py` — multiple `get_embedder` calls.

After Task 4 deletes the cloud fixtures, the LLM-side imports in
`test_governance_constraints.py` still need work. The constraints
themselves (`fitz_sage/governance/constraints/`) have their own
`EmbedderFunc` callable type and accept user-supplied embedders —
that's a deliberate extension point, preserved. But the integration
tests need to supply an embedder somehow (or be marked
`pytest.mark.skip` with a note).

Not urgent — these are skipped from the unit suite already.

---

### Task 9 — Optional: trim CHANGELOG.md cross-references

CHANGELOG.md still has hundreds of lines referencing the deleted
`fitz_sage/vector_db/`, `DiffIngestExecutor`, `HydeGenerator`,
embeddings, etc. Don't rewrite history (those entries are correct
records of past versions), but consider adding a top-of-file pointer
to the demolition commits so readers can orient.

---

### Task 10 — Replace PostgreSQL with SQLite

Decision made 2026-05-11. Postgres is heavy for a local-first RAG
library — server install, admin DB for `DROP DATABASE`, service-mode
config. SQLite + FTS5 covers the same retrieval semantics (full-text
search with native `bm25()` ranking) with zero install, file-based
storage, stdlib-only runtime. Strict simplification — same retrieval
architecture, smaller surface.

**Architectural Rule #3 needs updating** when this lands:
"PostgreSQL is the only storage" → "SQLite is the only storage."
Also update the TL;DR and Files/paths section of this HANDOFF.

#### Decisions to confirm before starting

1. **One DB file per collection** (analogous to current
   one-postgres-DB per collection — `delete_collection` becomes
   `os.unlink`). Recommend yes.
2. **FTS5 external-content** tables, so the original columns
   (`content`, `title`, `name`, `qualified_name`, `summary`) stay
   queryable for the parent-context joins
   `SectionSearchStrategy._score_results` does. Recommend yes.
3. **SQLite JSON1** (bundled in standard CPython `sqlite3`) for
   `json_extract`, replacing JSONB columns. Recommend yes.

#### Mapping

| Current (Postgres) | Target (SQLite) |
|---|---|
| `psycopg` + `PostgresConnectionManager` | stdlib `sqlite3` + new `SqliteConnectionManager` |
| `to_tsvector` GIN + `ts_rank_cd` | FTS5 virtual table + `MATCH` + `bm25()` |
| `CREATE DATABASE` (admin DB) + `DROP DATABASE` | one `<collection>.db` file; `unlink()` to delete |
| `JSONB` columns | `TEXT` + `json_extract` (JSON1) |
| `GENERATED ALWAYS AS (to_tsvector(...))` | FTS5 triggers, or external-content table |
| `pgserver` / postgres service | nothing — the file is the database |

#### Files to rewrite (high-level)

- `fitz_sage/storage/postgres.py` → `sqlite.py`. New
  `SqliteConnectionManager`, create/drop helpers, schema init.
- `fitz_sage/storage/__init__.py` — re-export the SQLite manager.
- `fitz_sage/engines/fitz_krag/ingestion/schema.py` — DDL for FTS5
  external-content tables over section/symbol/table indexes.
- `fitz_sage/engines/fitz_krag/ingestion/{section,symbol,table}_store.py`
  — `search_bm25` becomes `MATCH ... ORDER BY bm25(...)`; upserts
  become `INSERT ... ON CONFLICT DO UPDATE`.
- `fitz_sage/engines/fitz_krag/retrieval/strategies/section_search.py`
  — port the `_score_results` parent-context join. RRF logic
  unchanged.
- `fitz_sage/engines/fitz_krag/engine.py`, `runtime.py`,
  `services/fitz_service.py` — swap connection manager; rewrite
  `list_collections` to scan the data dir for `*.db`,
  `delete_collection` to `os.unlink`.
- `fitz_sage/retrieval/vocabulary/store.py`,
  `fitz_sage/retrieval/entity_graph/store.py`,
  `fitz_sage/retrieval/sparse/index.py` — port.
- `fitz_sage/tabular/store/postgres.py` → `sqlite.py`.
- `fitz_sage/core/detect.py` — drop postgres detection entirely.
  SQLite is stdlib; nothing to probe.
- `fitz_sage/cli/{context,services/init_service,commands/init_wizard,commands/config}.py`
  — drop postgres UI/detection. Replace postgres connection knobs
  with a single `storage_path` field.
- `tests/unit/test_postgres_connection.py`,
  `test_postgres_recovery.py`, `test_postgres_table_store.py` —
  rename and rewrite.
- `tests/conftest.py`, `tests/unit/conftest.py` — drop postgres
  fixtures, add sqlite tempdir fixtures.
- `pyproject.toml` — drop `psycopg`, `pgserver`. Rename or remove the
  `postgres:` pytest marker (line 185).

#### Verification

```bash
python -m pytest tests/unit
python .smoke_test/smoke_test.py
```

Same baseline expected (5/5 answered, 6/6 substrings, 3/5 mode-match) —
retrieval semantics preserved.

#### Cleanup after migration lands

- Update Architectural Rule #3 and the TL;DR in HANDOFF.
- Update the Files/paths-to-know section in HANDOFF.
- Add a CHANGELOG entry for the storage swap.
- **Task 6 docs refresh must wait until this lands** — don't write
  docs for an architecture about to change.

---

## How to verify the whole thing works

After each task, run:

```bash
# 1. Unit suite green
python -m pytest tests/unit -q --ignore=tests/unit/integrations

# 2. End-to-end smoke against LM Studio (port 1234, smoke-chat loaded)
curl -s http://localhost:1234/v1/models | python -m json.tool
python .smoke_test/smoke_test.py

# 3. Production imports clean
python -c "from fitz_sage.engines.fitz_krag.engine import FitzKragEngine; \
           from fitz_sage.cli.context import CLIContext; \
           from fitz_sage.services.fitz_service import FitzService; \
           print('imports OK')"
```

The smoke baseline today is: 5/5 queries answered, 6/6 expected
substrings, 3/5 governance mode-match (the 2 misses require
`enable_guardrails=True`, which the smoke disables for speed).

## Key context: the user's actual workflow

- **OS:** Windows. PowerShell shell. Bash also available via the Bash
  tool. Don't suggest `apt-get` or assume `/usr/bin`.
- **VRAM:** RTX 5090, 32 GB. Models loaded via LM Studio CLI (`lms`).
  Verify what's loaded with `lms ls` / `lms ps`; the smoke test
  requires the identifier `smoke-chat` to be loaded.
- **Other projects:** The user is also working on
  `expert-prefetch-v3` (TEC, llama.cpp MoE inference) and
  `fitz-forge`. fitz-sage is one of several related projects; don't
  conflate them.
- **Tooling:** `rtk` is the user's token-saving CLI wrapper.
  Project-level `CLAUDE.md` (read it) requires prefixing commands
  with `rtk` for git/grep/etc. Standard tools (Glob, Grep, Read,
  Edit) are fine without `rtk`.

## Files / paths to know

- `fitz_sage/engines/fitz_krag/engine.py` — the heart. ~1500 LOC.
- `fitz_sage/engines/fitz_krag/config/schema.py` — `FitzKragConfig`.
  Currently 21 fields; the legacy weight + vector_db + embedding
  fields have all been pruned.
- `fitz_sage/engines/fitz_krag/retrieval/strategies/` — three files
  (`code_search.py`, `section_search.py`, `table_search.py`) plus
  `llm_code_search.py`. Each `retrieve()` takes
  `(query, limit, detection=None)` — no embedder, no query_vector,
  no hyde_vectors.
- `fitz_sage/engines/fitz_krag/retrieval/router.py` — `RetrievalRouter`.
  Constructor:
  `(code_strategy, config, section_strategy=None, table_strategy=None,
  chat_factory=None, agentic_strategy=None)`. No embedder, no
  hyde_generator, no chunk_strategy.
- `fitz_sage/engines/fitz_krag/retrieval_profile.py` — `RetrievalProfile`
  + `build_retrieval_profile`. No `run_hyde`, no `fallback_to_chunks`
  fields.
- `fitz_sage/engines/fitz_krag/ingestion/schema.py` —
  `ensure_schema(connection_manager, collection)` (no `embedding_dim`).
  All three index tables have `content_tsv` GIN indexes; no `vector`
  columns.
- `fitz_sage/llm/client.py` — `get_chat`, `get_reranker`,
  `get_vision`. No `get_embedder`.
- `fitz_sage/llm/providers/openai_compat.py` — `OpenAICompatChat` and
  `OpenAICompatVision`. No `OpenAICompatEmbedding`.
- `fitz_sage/llm/providers/llm_reranker.py` — `LLMReranker`.
- `fitz_sage/governance/decider.py` — v6 cascade ML classifier.
  Requires `xgboost` and `lightgbm`.
- `tests/e2e_krag/fixtures_rag/` — small mixed corpus (~9 files)
  used by the smoke test. Includes a known conflict between Finance
  and HR on TechCorp employee count (5,200 vs 4,800), useful for
  testing DISPUTED governance.
- `.smoke_test/smoke_test.py` — end-to-end test harness against
  fixtures_rag using LM Studio at port 1234. Single-mode after
  `42fd4c7d`.
- `.smoke_test/results.json` — last run's results.

## Conversation context for the next session

This session ripped out the entire dense-retrieval scaffolding the
prior session had left behind:

1. Tasks 1+2+3 from the original handoff plus an unplanned
   mid-Task-1 architectural cleanup. The original handoff's order
   was right; what got added was a "no shims" pass between Task 1's
   test-greening and Task 2.
2. The user introduced a durable feedback rule mid-session:
   *"are you fixing everything around the tests or do you actually
   check if what you fix makes sense? the tests are not the truth!"*
   and *"are you leaving legacy shims? or are you just doing
   correct architectural edits only?"* — both saved to auto-memory.
   The new project rule (Architectural rule #4 above) reflects this.
3. The smoke test ran clean end-to-end at every checkpoint. The
   3/5 mode-match baseline is stable.

The repository is in a much smaller state than the original handoff
described. Total deletions from this session: roughly **11k lines
removed, 300 lines added** across the four commits.

## Final thing — don't waste a context window second-guessing

The architecture is committed. The user evaluated and chose. Next
session: execute Task 4 (cloud delete), then Task 10 (postgres →
sqlite — the big one), then Task 6 (docs, over the post-migration
architecture), then Task 7 (enrichment, if the user OKs), then
Task 8 (integration test cleanup). Run the verification commands
after each. Commit each task as its own commit with a clear message.
Don't ask "are you sure about no embeddings?" — they're sure.
Don't ask "are you sure about sqlite?" — they are, as of 2026-05-11.

## Code Generation Rules (MANDATORY)

1. **Never alter unrelated code** - Touch only code directly related to the requested change
2. **File path comment required** - First line: `# fitz_sage/engines/fitz_krag/engine.py`
3. **No legacy code** - No backwards compatibility, no deprecated code, no shims. Delete completely when removing
4. **Tests follow architecture** - Fix tests to match new architecture, never compromise code quality for tests
5. **Always use .venv for pip** - `.venv/Scripts/pip install <package>` (Windows) or `.venv/bin/pip install <package>` (Unix)

## Project Overview

**fitz-sage** - Local-first, modular RAG knowledge engine platform with epistemic honesty and full provenance.

```bash
pip install -e ".[dev]"   # Install for development
pytest                    # Run tests
black . && isort .        # Format code
python -m tools.contract_map  # Check architecture
```

## Architecture

```
fitz_sage/
├── core/          # Paradigm-agnostic (Query, Answer, Provenance, Constraints)
├── engines/fitz_krag/   # KRAG: retrieval/, generation/, ingestion/
├── retrieval/     # SHARED intelligence (detection, entity_graph, rewriter)
├── ingestion/     # Parser → Chunking
├── llm/           # Chat / LLM reranker providers — single OpenAI-compatible HTTP protocol
├── storage/       # SQLite connection manager (one .db per collection, FTS5 + bm25)
├── tabular/       # CSV/table query with SQL generation (SqliteTableStore)
├── runtime/       # Multi-engine orchestration
├── cli/           # commands/; api/; sdk/
```

**Layer dependencies** (enforced by `contract_map`):
- `core/` ← no imports from engines/, ingestion/
- `engines/` ← core/, llm/, storage/, retrieval/
- `retrieval/`, `llm/`, `ingestion/` ← core/ only
- `runtime/`, `cli/` ← everything

**Engine protocol**: all engines implement `answer(query: Query) -> Answer` from `core/engine.py`.

## Code Style

- **Black** (line-length 100), **isort** (black profile)
- **Type hints** required for public APIs; **Docstrings**: Google style
- `snake_case` modules/functions, `PascalCase` classes, `UPPER_SNAKE` constants

## Developer Tools (use these, not find/grep/cat)

| Task | Tool |
|------|------|
| Find files | `fd <pattern>` |
| Search text | `rg <pattern>` |
| Search code structure | `ast-grep --pattern '<code>'` |
| Symbol definitions | `ctags -R` then `grep "Symbol" tags` |
| Codebase stats | `tokei` / `tokei <dir>` |
| JSON/YAML | `jq '.<key>' file.json` / `yq '.<key>' file.yaml` |
| CSV | `xsv headers`, `xsv count`, `xsv stats` |
| Git diffs | `git diff \| delta` |

## Testing

```bash
pytest                      # All tests
pytest -m "not slow"        # Skip slow
pytest --cov=fitz_sage        # With coverage
pytest tests/unit/          # Unit only
```

**Markers**: `slow`, `integration`, `e2e`, `performance`, `security`, `scalability`, `chaos`

## Design Principles

1. **Explicit over clever** - No magic, config-driven
2. **Honest over helpful** - "I don't know" > hallucination
3. **No `enabled` flags** - Provider presence IS the feature toggle
4. **All retrieval intelligence is baked in** - not configured

## Feature Control

Config declares WHAT provider; plugin choice determines IF feature is used:
- `rerank: onnx` → reranking enabled; `rerank: null` → disabled
- `parser: "docling_vision"` → uses VLM from `vision:` config; `parser: "docling"` → no VLM

## Retrieval Intelligence (automatic, not configured)

Temporal, query expansion, BM25, multi-query, comparison handling, entity graph, freshness boosting, aggregation detection, multi-hop, reranking.

**Core files**:
- `engines/fitz_krag/retrieval/router.py` - dispatches strategies, fuses and ranks results
- `engines/fitz_krag/query_batcher.py` - one batched LLM call (analysis + detection + rewriting + keywords)
- `retrieval/detection/modules/` - temporal, aggregation, comparison, freshness
- `retrieval/detection/detectors/expansion.py` - dict-based synonym/acronym expansion (not LLM)

## Extensibility (DO NOT create parallel implementations)

| Need | Extend | NOT |
|------|--------|-----|
| Detect query patterns | `retrieval/detection/modules/` (new `DetectionModule`) | Inline regex in the router |
| Chunk metadata at ingestion | `engines/fitz_krag/ingestion/enricher.py` (`KragEnricher`) | Post-hoc at query time |
| Add synonyms/acronyms | `detection/detectors/expansion.py` dicts | New expander class |

**Detection module pattern** (LLM-based): implement `category`, `json_key`, `prompt_fragment()`, `parse_result()` → add to `DEFAULT_MODULES` in `modules/__init__.py`. All modules combine into one LLM call (via `QueryBatcher`).

**Ingestion pipeline**: `KragIngestPipeline` runs per file — `parse_file` (symbols/sections/tables, no LLM) → `summarize_file` (LLM summaries) → `enrich_file` (`KragEnricher`: keywords + entities; L1 hierarchy summary) — then a corpus `finalize` (import graph + L2 hierarchy summary). `enable_enrichment` / `enable_hierarchy` gate the LLM stages.

## Key Files

| Purpose | Path |
|---------|------|
| Engine protocol | `core/engine.py` |
| KRAG engine | `engines/fitz_krag/engine.py` |
| Config schema | `engines/fitz_krag/config/schema.py` |
| Config loader | `config/loader.py` |
| Query-prep batcher | `engines/fitz_krag/query_batcher.py` |
| Governance classifier | `governance/pyrrho.py` |
| KRAG enricher | `engines/fitz_krag/ingestion/enricher.py` |
| Multi-hop | `engines/fitz_krag/retrieval/multihop.py` |
| Parser routing | `ingestion/parser/router.py` |

## Configuration System

Layered merge: package defaults → `~/.fitz/config/<engine>.yaml` (user overrides).

Sections: `chat_fast`, `chat_balanced`, `chat_smart`, `chat_base_url`, `chat_api_key_env`, `rerank`, `retrieval`, `chunking`. (Pre-v0.12.0 `embedding`, `vector_db`, `vector_db_kwargs`, `cloud` are gone.)

Storage: SQLite, one `.db` per collection under `<workspace>/sqlite/`. No server, no `connection_string` knob.

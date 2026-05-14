# Engines

fitz-sage **v0.12.0+**. An engine is anything that implements the
`KnowledgeEngine` protocol — given a `Query`, return an `Answer` with
mode (`TRUSTWORTHY` / `DISPUTED` / `ABSTAIN`) and provenance.

The shipping engine is `fitz_krag` (Knowledge Routing Augmented
Generation). It's the only one most users need.

---

## Core Contracts

Defined in `fitz_sage/core/`:

### `KnowledgeEngine`

```python
from typing import Protocol
from fitz_sage.core import Query, Answer

class KnowledgeEngine(Protocol):
    def answer(self, query: Query) -> Answer: ...
```

### `Query`

```python
@dataclass
class Query:
    text: str
    constraints: Constraints | None = None
    metadata: dict | None = None
```

### `Answer`

```python
@dataclass
class Answer:
    text: str
    mode: AnswerMode                  # TRUSTWORTHY | DISPUTED | ABSTAIN
    provenance: list[Provenance]
    metadata: dict
```

### `Provenance`

```python
@dataclass
class Provenance:
    source_id: str        # collection-qualified address
    excerpt: str | None
    metadata: dict
```

---

## `fitz_krag` — the production engine

**Location:** `fitz_sage/engines/fitz_krag/`

KRAG is a structure-first retriever. Instead of chunking documents into
fixed-size text windows and embedding them, it parses code, prose, and
tables into typed units (`Symbol`, `Section`, `TableSpec`) and routes
queries to the right strategy.

```
Query
 ├─► Rewriter (resolves pronouns / coreference via chat call)
 ├─► Analyzer (detects intent: temporal, comparison, aggregation, ...)
 ├─► Router (symbol search · section search · table SQL)
 │    └─► FTS5 + bm25() over per-collection .db
 ├─► Expander (import graph, entity links, same-file refs, hierarchy)
 ├─► OnnxReranker (ONNX cross-encoder, ~30 ms CPU)
 ├─► Constraints (conflict_aware, insufficient_evidence, ...)
 └─► Synthesizer → Answer (+ provenance + mode)
```

### Usage

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
answer = engine.answer(Query(text="What is X?"))
```

The convenience function `run_fitz_krag(text, **overrides)` wraps
this for one-shots.

### Configuration

See [CONFIG.md](CONFIG.md) for every key. The minimum is `collection:`
and a chat tier. Everything else has working defaults.

### Built-in features

| Feature                 | What it does                                                  |
| ----------------------- | ------------------------------------------------------------- |
| Symbol / section / table routing | Per-content-type retrieval strategies              |
| Import graph traversal  | Code: walks references and imports across files               |
| Entity linking          | Cross-source linking via shared named entities                |
| Hierarchical summaries  | L1 (section), L2 (doc-level) summaries built at ingest        |
| Multi-hop retrieval     | Iterative bridge extraction for compound questions            |
| ONNX reranker           | INT8 cross-encoder, single forward pass on CPU                |
| Epistemic guardrails    | TRUSTWORTHY / DISPUTED / ABSTAIN constraint cascade           |
| Artifact generation     | Architecture narrative, dependency summary, etc. per collection |
| Incremental ingestion   | Re-ingest only changed files (`.fitz/ingest_state.json`)      |

---

## Engine selection

The default engine is `fitz_krag`. Choose another via the runtime API
or the CLI:

```python
from fitz_sage import run

answer = run("What is X?", engine="fitz_krag")
```

```bash
fitz query "What is X?" --engine fitz_krag --source ./docs
```

### Available engines

| Engine     | Status     | Description                                         |
| ---------- | ---------- | --------------------------------------------------- |
| `fitz_krag`| Production | KRAG with epistemic guardrails (this doc's subject) |

Custom engines register through the engine registry — see
[CUSTOM_ENGINES.md](CUSTOM_ENGINES.md).

---

## Custom engines

```python
from fitz_sage.core import Query, Answer, AnswerMode
from fitz_sage.runtime import EngineRegistry

class MyEngine:
    def answer(self, query: Query) -> Answer:
        return Answer(
            text="...",
            mode=AnswerMode.TRUSTWORTHY,
            provenance=[],
            metadata={},
        )

EngineRegistry.get_global().register("my_engine", lambda cfg: MyEngine())
```

You don't need to subclass anything — duck-typing on the protocol is
enough. See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md) for the registry and
config-loader hooks.

---

## Standalone code retrieval

For code-only use cases where ingesting a full collection is overkill,
fitz-sage ships a lightweight `CodeRetriever` that reads files
directly from disk — no SQLite, no ingestion pipeline:

```bash
pip install fitz-sage[code]
```

```python
from fitz_sage.code import CodeRetriever
from fitz_sage.llm.factory import get_chat_factory

retriever = CodeRetriever(
    source_dir="./myproject",
    chat_factory=get_chat_factory({
        "fast":   "endpoint",
        "smart":  "endpoint",
    }, base_url="http://localhost:8080/v1", smart_model="qwen2.5-7b-instruct"),
)
results = retriever.retrieve("How does authentication work?")
```

Pipeline: AST structural index → LLM file selection → import-graph
expansion → neighbor-directory expansion → compression. No database.

| Component                          | Path                          |
| ---------------------------------- | ----------------------------- |
| `CodeRetriever`                    | `fitz_sage/code/retriever.py` |
| Indexer (file list, AST, imports)  | `fitz_sage/code/indexer.py`   |
| LLM prompts                        | `fitz_sage/code/prompts.py`   |

---

## Architecture principles

1. **Protocol over inheritance.** Implement `KnowledgeEngine` by
   exposing a single `answer()` method.
2. **Config-driven.** Engine behaviour lives in YAML / `*Config`
   dataclasses, not in Python keywords.
3. **Shared infrastructure.** Chat layer, SQLite storage, and
   ingestion are shared across engines.
4. **Honest answers.** Every `Answer` carries a `mode` — engines never
   pretend they know something they don't.

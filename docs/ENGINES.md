<!-- docs/ENGINES.md -->
# Engines

The base engine protocol still supports `answer(Query) -> Answer`, but the
production `fitz_krag` engine is
retrieval-first: use `evidence(Query) -> EvidencePack` when you want the
ranked source material without generation.

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
    metadata: dict = field(default_factory=dict)
```

### `Answer`

```python
@dataclass
class Answer:
    text: str
    mode: AnswerMode | None           # runtime: SUFFICIENT | DISPUTED | INSUFFICIENT
    provenance: list[Provenance]
    metadata: dict
```

### `Provenance`

```python
@dataclass
class Provenance:
    source_id: str        # stable engine-specific source identifier
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
 ├─► Query prep (deterministic signals, explicit clauses, managed Qwen keywords)
 ├─► Optional query intelligence (rewrite / analyze / detect)
 ├─► Router (symbol search · section search · table metadata)
│    └─► FTS5 + bm25() over per-collection .db
 ├─► OnnxReranker (bounded INT8 ONNX cross-encoder)
 ├─► Read + bounded evidence closure + compilation
 ├─► Progressive Pyrrho prefixes (3, then +2 while INSUFFICIENT)
 ├─► EvidencePack
 └─► Optional synthesizer → Answer (+ provenance + mode)
```

### Usage

```python
from pathlib import Path

from fitz_sage.core import Query
from fitz_sage.engines.fitz_krag.config.schema import FitzKragConfig
from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

cfg = FitzKragConfig(
    collection="my_docs",
)
engine = FitzKragEngine(cfg)
engine.point(Path("./docs"), start_worker=False)
pack = engine.evidence(Query(text="What is X?"))
```

### Retrieval-first evidence

`evidence()` returns the governed, serializable evidence pack. This is the
contract used by `fitz retrieve`:

```python
pack = engine.evidence(Query(text="Where is auth handled?"))
for item in pack.items:
    print(item.file_path, item.excerpt)
```

For raw engine-specific source objects without governance packaging, call
`retrieve()`:

```python
results = engine.retrieve(Query(text="Where is auth handled?"))
for r in results:                       # list[ReadResult]
    print(r.file_path, r.content)
```

`retrieve()` returns `ReadResult`s (`file_path`, `content`, and an `Address`
with provenance and score), skipping evidence packaging and optional synthesis.

### Configuration

See [CONFIG.md](CONFIG.md) for every key. The minimum is `collection:`.
Chat providers are optional and only needed for synthesized answers, optional
query intelligence, or vision parsing. Managed Qwen query terms and optional
background enrichment are internal.

### Built-in features

| Feature                 | What it does                                                  |
| ----------------------- | ------------------------------------------------------------- |
| Symbol / section / table routing | Per-content-type retrieval strategies              |
| Import graph traversal  | Code: walks references and imports across files               |
| Entity linking          | Cross-source linking via shared named entities                |
| Hierarchical summaries  | L1 file summaries and L2 corpus overview built during enrichment |
| Evidence closure        | Bounded bridge retrieval for unresolved query obligations      |
| ONNX reranker           | Bounded INT8 cross-encoder, two batch-one CPU workers          |
| Epistemic governance    | Pyrrho v2 sufficient / disputed / insufficient evidence verdicts |
| Source indexing         | Parse and persist before `point()` returns; enrich afterward |

---

## Engine selection

The default engine is `fitz_krag`. Choose another via the runtime API
or the CLI:

```python
from fitz_sage import run

# run() calls answer(), so fitz_krag requires a configured synthesizer here.
answer = run("What is X?", engine="fitz_krag")
```

```bash
fitz retrieve "What is X?" --engine fitz_krag --source ./docs
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
from fitz_sage.core import Answer, Query
from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.runtime import EngineRegistry

class MyEngine:
    def answer(self, query: Query) -> Answer:
        return Answer(
            text="...",
            mode=AnswerMode.SUFFICIENT,  # runtime mode for sufficient evidence
            provenance=[],
            metadata={},
        )

EngineRegistry.get_global().register("my_engine", lambda config: MyEngine())
```

You don't need to subclass anything — duck-typing on the protocol is
enough. See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md) for the registry and
config-loader hooks.

---

## Architecture principles

1. **Protocol over inheritance.** Implement `KnowledgeEngine` by
   exposing a single `answer()` method.
2. **Config-driven.** Engine behaviour lives in YAML / `*Config`
   dataclasses, not in Python keywords.
3. **Optional shared infrastructure.** Engines may reuse Fitz-Sage chat,
   storage, and ingestion modules, but the core protocol does not require them.
4. **Honest evidence.** Every `EvidencePack` carries a `mode`; optional
   answers inherit that epistemic posture.

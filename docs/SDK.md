# docs/SDK.md

Complete reference for the Fitz Python SDK.

---

## Quick Start

```python
import fitz_sage

pack = fitz_sage.evidence("What is the refund policy?", source="./docs")
print(pack.mode)
for item in pack.items:
    print(item.file_path, item.excerpt)
```

---

## Module-Level API

The simplest retrieval-first SDK call is `fitz_sage.evidence(...)`. It matches
the default `fitz query` CLI behavior.

### fitz_sage.evidence()

Retrieve a governed evidence pack without answer synthesis.

```python
fitz_sage.evidence(
    question: str,                 # The question to retrieve evidence for
    source: str | Path = None,     # If provided, registers documents first
    collection: str = None,        # Collection name (uses default if not specified)
) -> EvidencePack
```

**Returns:** `EvidencePack` with `query`, `mode`, ranked `items`, `reasons`,
`timings`, `indexing_status`, and `metadata`.

**Examples:**

```python
pack = fitz_sage.evidence("What is the refund policy?", source="./docs")
print(pack.mode)  # runtime AnswerMode: SUFFICIENT, DISPUTED, INSUFFICIENT, or None

for item in pack.items:
    print(f"{item.rank}. {item.file_path}:{item.line_range}")
    print(item.excerpt)
```

### fitz_sage.trace()

Return a versioned record of the same governed retrieval execution.

```python
run = fitz_sage.trace(
    "What is the refund policy?",
    source="./docs",
    collection="policies",
)

run.write("run.json")  # source content redacted
run.write("run-with-content.json", include_content=True)
print(run.explain())
```

Use `fitz_sage.replay_governance(...)` to evaluate another governance provider
over a content-bearing record's frozen evidence:

```python
result = fitz_sage.replay_governance(
    "run-with-content.json",
    governance="pyrrho/C:/models/pyrrho-candidate",
)
```

See [Retrieval Execution Records](RETRIEVAL_RUNS.md) for security and replay
semantics.

### fitz_sage.query()

Generate a synthesized answer. This requires a configured synthesizer provider;
use `evidence()` for the default retrieval path.

```python
fitz_sage.query(
    question: str,                 # The question to ask
    source: str | Path = None,     # If provided, registers documents before querying
    collection: str = None,        # Collection name (uses default if not specified)
) -> Answer
```

**Returns:** `Answer` with `text`, `provenance`, `mode`

**Examples:**

```python
answer = fitz_sage.query("What is the refund policy?")
print(answer.text)
print(answer.mode)  # runtime AnswerMode: SUFFICIENT, DISPUTED, or INSUFFICIENT

# Access sources
for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

---

## fitz Class

For advanced usage with multiple collections or custom configuration.

### Constructor

```python
from fitz_sage import fitz

f = fitz(
    collection: str = "default",     # Collection name
    config_path: str = None,         # Custom config file
    auto_init: bool = True           # Create config if missing
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `collection` | str | `"default"` | Collection name |
| `config_path` | str/Path | None | Path to YAML config |
| `auto_init` | bool | True | Create default config if missing |

### Methods

#### query()

```python
f.query(
    question: str,
    source: str | Path = None,  # If provided, registers documents before querying
) -> Answer  # synthesized answer: text, provenance, mode
```

Requires `synthesizer:` in config or an engine instance configured with a
synthesizer provider.

#### evidence()

```python
f.evidence(
    question: str,
    source: str | Path = None,  # If provided, registers documents before retrieval
) -> EvidencePack
```

#### trace()

```python
f.trace(
    question: str,
    source: str | Path = None,
) -> RetrievalRun

f.replay_governance(
    run: RetrievalRun | str | Path,
    governance: str | object = None,
) -> GovernanceReplay
```

#### point()

Register a source file or directory. Indexing runs in the background — queries
work after the parsed search surface is ready and improve as Qwen enrichment
completes.

```python
f.point(source: str | Path) -> None
```

#### retrieve()

The raw sources behind an answer/evidence pack, without governance packaging.
For KRAG, returns `ReadResult` objects with `content`, `file_path`, and
`line_range`. Most applications should prefer `evidence()`.

```python
f.retrieve(question: str) -> list
```

#### wait_for_query_surface() / wait_for_indexing() / indexing_status()

```python
f.wait_for_query_surface() -> None  # block until parsed units are searchable
f.wait_for_indexing() -> None       # block until query-ready keywording completes
f.indexing_status() -> dict         # query-ready + deep-enrichment progress
```

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `collection` | str | The collection name |
| `config_path` | Path | Path to config file |

### Examples

```python
from fitz_sage import fitz

# Multiple collections
physics = fitz(collection="physics")
physics_pack = physics.evidence("Explain entanglement", source="./physics_papers")

legal = fitz(collection="legal")
legal_pack = legal.evidence("What are the payment terms?", source="./contracts")

# Custom config
f = fitz(config_path="./my_config.yaml")

# Require existing config
f = fitz(auto_init=False)  # Raises if no config
```

---

## Core Types

### Answer

The response from a query.

```python
from fitz_sage import Answer

class Answer:
    text: str                    # The answer text
    provenance: list[Provenance] # Sources used
    mode: AnswerMode | None      # Epistemic mode
    metadata: dict               # Additional data
```

**Runtime Answer Modes:**

Pyrrho v2's native model verdict is available in governance metadata as
`evidence_verdict`. `AnswerMode` is the runtime API value.

| Mode | Description |
|------|-------------|
| `SUFFICIENT` | Runtime mode for `SUFFICIENT` evidence |
| `DISPUTED` | Runtime mode for `DISPUTED` evidence |
| `INSUFFICIENT` | Runtime mode for `INSUFFICIENT` evidence |

### EvidencePack

The retrieval-first response contract.

```python
from fitz_sage import EvidencePack, EvidenceItem

class EvidencePack:
    query: str
    mode: AnswerMode | None
    items: list[EvidenceItem]
    reasons: list[str]
    timings: dict
    indexing_status: dict
    metadata: dict

class EvidenceItem:
    rank: int
    source_id: str
    file_path: str
    address_kind: str
    address_location: str
    line_range: tuple[int, int] | None
    score: float | None
    excerpt: str
    content: str
    metadata: dict
```

Use `pack.to_dict()` or `pack.to_json()` for API responses and downstream apps.

### RetrievalRun

An inspectable, versioned record containing the effective query plan, term
origins, retrieval strategies, candidate stages, compiled evidence ranking,
governance trajectory, selected `EvidencePack`, and runtime fingerprints.

`to_dict()`, `to_json()`, and `write()` redact source bodies by default.
Content-bearing traces are required for governance replay.

### Provenance

Source attribution for an answer.

```python
from fitz_sage import Provenance

class Provenance:
    source_id: str    # Unique source identifier
    excerpt: str      # Relevant excerpt
    metadata: dict    # Additional source info
```

### Query

Input to the engine (for advanced usage).

```python
from fitz_sage import Query

query = Query(
    text="What is X?",
    metadata={"user_id": "123"}
)
```

---

## Advanced Usage

### Direct Engine Access

`create_engine()` returns an engine implementing the `RetrievalEngine` protocol —
`evidence()`, optional `answer()`, and the full ingest/retrieve lifecycle:

```python
from pathlib import Path
from fitz_sage import create_engine, Query

engine = create_engine("fitz_krag")
engine.load("default")                  # bind to a collection
engine.point(Path("./docs"))            # register a source
engine.wait_for_query_surface()         # parsed units are searchable

pack = engine.evidence(Query(text="What is X?"))     # governed evidence
answer = engine.answer(Query(text="What is X?"))     # synthesized answer
sources = engine.retrieve(Query(text="What is X?"))  # raw sources, no governance packaging
```

### Engine Selection

```python
from fitz_sage import run, list_engines

# List available engines
engines = list_engines()
print(engines)  # ['fitz_krag']

# Run with specific engine
answer = run("What is X?", engine="fitz_krag")
```

### Fitz KRAG Specific

```python
from fitz_sage.engines.fitz_krag.runtime import run_fitz_krag

# KRAG-specific entry point
answer = run_fitz_krag("What is X?")

# Create a reusable KRAG engine via the canonical factory
from fitz_sage import create_engine

engine = create_engine("fitz_krag")
```

---

## Error Handling

```python
from fitz_sage import (
    ConfigurationError,
    EngineError,
    QueryError,
    KnowledgeError,
    GenerationError,
)

try:
    pack = fitz_sage.evidence("What is X?")
except ConfigurationError as e:
    print(f"Config issue: {e}")
except QueryError as e:
    print(f"Query failed: {e}")
except EngineError as e:
    print(f"Engine error: {e}")
```

| Exception | When |
|-----------|------|
| `ConfigurationError` | Config file missing or invalid |
| `QueryError` | Invalid query or retrieval failed |
| `EngineError` | Engine initialization or execution error |
| `GenerationError` | LLM generation failed |
| `KnowledgeError` | Base class for knowledge errors |

---

## Configuration

The SDK uses the same config as CLI. See [CONFIG.md](CONFIG.md) for details.

**Config search order:**
1. `config_path` parameter (if provided)
2. `.fitz/config.yaml` in the current workspace
3. Auto-created default config (if `auto_init=True`)

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [API.md](API.md) - REST API documentation
- [INGESTION.md](INGESTION.md) - Ingestion pipeline details

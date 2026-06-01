# docs/SDK.md

Complete reference for the Fitz Python SDK (v0.14.1).

---

## Quick Start

```python
import fitz_sage

answer = fitz_sage.query("What is the refund policy?", source="./docs")
print(answer.text)
```

---

## Module-Level API

The simplest way to use Fitz - matches CLI behavior.

### fitz_sage.query()

Query the knowledge base.

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
print(answer.mode)  # TRUSTWORTHY, DISPUTED, or ABSTAIN

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

#### point()

Register a source file or directory. Indexing runs in the background — queries
work immediately and improve as it completes.

```python
f.point(source: str | Path) -> None
```

#### retrieve()

The raw sources behind an answer, without synthesis (useful for building your
own citations or synthesis). For KRAG, returns `ReadResult` objects with
`content`, `file_path`, and `line_range`.

```python
f.retrieve(question: str) -> list
```

#### wait_for_indexing() / indexing_status()

```python
f.wait_for_indexing() -> None   # block until background indexing completes
f.indexing_status() -> dict     # {total, indexed, pending, complete, by_state}
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
physics_answer = physics.query("Explain entanglement", source="./physics_papers")

legal = fitz(collection="legal")
legal_answer = legal.query("What are the payment terms?", source="./contracts")

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

**Answer Modes:**

| Mode | Description |
|------|-------------|
| `TRUSTWORTHY` | Strong evidence supports the answer |
| `DISPUTED` | Conflicting sources detected |
| `ABSTAIN` | Insufficient evidence to answer |

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
`answer()` plus the full ingest/retrieve lifecycle:

```python
from pathlib import Path
from fitz_sage import create_engine, Query

engine = create_engine("fitz_krag")
engine.load("default")                  # bind to a collection
engine.point(Path("./docs"))            # register a source (indexes in background)

answer = engine.answer(Query(text="What is X?"))     # synthesized answer
sources = engine.retrieve(Query(text="What is X?"))  # raw sources, no synthesis
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
    answer = fitz_sage.query("What is X?")
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
2. `~/.fitz/config/fitz_krag.yaml` (user config)
3. Auto-created default config (if `auto_init=True`)

---

## See Also

- [CONFIG.md](CONFIG.md) - Configuration reference
- [API.md](API.md) - REST API documentation
- [INGESTION.md](INGESTION.md) - Ingestion pipeline details

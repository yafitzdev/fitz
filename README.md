# Fitz v0.3.0 - Engine Architecture

## What's New in v0.3.0

Fitz has been restructured as an **engine platform** rather than a RAG framework. This enables:

- ✅ **Clean abstractions**: Query → Engine → Answer (paradigm-agnostic)
- ✅ **Multiple paradigms**: Classic RAG today, CLaRa tomorrow, custom engines anytime
- ✅ **Forward compatible**: Your code won't break when new engines are added
- ✅ **Better organization**: Engines are self-contained, infrastructure is shared

---

## Quick Start

### Installation

```bash
pip install fitz
```

### Basic Usage (New API)

```python
from fitz.engines.classic_rag import run_classic_rag

# Simple query
answer = run_classic_rag("What is quantum computing?")
print(answer.text)

# View sources
for source in answer.provenance:
    print(f"Source: {source.source_id}")
    print(f"Excerpt: {source.excerpt}")
```

### Advanced Usage

```python
from fitz.core import Query, Constraints
from fitz.engines.classic_rag import create_classic_rag_engine

# Create reusable engine
engine = create_classic_rag_engine("config.yaml")

# Query with constraints
query = Query(
    text="Explain quantum entanglement",
    constraints=Constraints(
        max_sources=5,
        filters={"topic": "physics"}
    )
)

answer = engine.answer(query)
```

---

## Architecture

### Before v0.3.0 (RAG Framework)
```
fitz/
├── pipeline/      # RAG-specific
├── retrieval/     # RAG-specific
├── generation/    # RAG-specific
└── core/          # Mixed concerns
```

### After v0.3.0 (Engine Platform)
```
fitz/
├── core/                  # Paradigm-agnostic contracts
│   ├── engine.py          # KnowledgeEngine protocol
│   ├── query.py           # Query type
│   └── answer.py          # Answer type
├── engines/
│   ├── classic_rag/       # RAG implementation
│   │   ├── engine.py      # ClassicRagEngine
│   │   ├── runtime.py     # Canonical entry point
│   │   ├── pipeline/      # RAG pipeline logic
│   │   ├── retrieval/     # RAG retrieval
│   │   └── generation/    # RAG generation
│   └── clara/             # Future: CLaRa engine
├── llm/                   # Shared LLM service
├── vector_db/             # Shared vector DB service
└── ingest/                # Shared ingestion
```

**Key principle:** Knowledge → Engine → Answer

---

## CLI

### Query Command

```bash
# Basic query
fitz-pipeline query "What is quantum computing?"

# With constraints
fitz-pipeline query "Explain X" --max-sources 5

# With filters
fitz-pipeline query "What is Y?" --filters '{"topic": "physics"}'

# With preset
fitz-pipeline query "Analyze this" --preset local
```

### Config Command

```bash
# Show resolved configuration
fitz-pipeline config show

# Show with custom config
fitz-pipeline config show --config my_config.yaml
```

---

## Migration Guide (v0.2.x → v0.3.0)

### Old Code (v0.2.x)
```python
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.pipeline.config.loader import load_config

config = load_config()
pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")

print(result.answer)
for source in result.sources:
    print(source.chunk_id)
```

### New Code (v0.3.0)
```python
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is X?")

print(answer.text)
for source in answer.provenance:
    print(source.source_id)
```

### Breaking Changes

1. **Import paths changed**: `fitz.pipeline.*` → `fitz.engines.classic_rag.*`
2. **Answer format changed**: `RGSAnswer.answer` → `Answer.text`
3. **Source format changed**: `sources` → `provenance`
4. **Entry point changed**: `RAGPipeline.run()` → `run_classic_rag()`

### Backwards Compatibility

The old `RAGPipeline` API still works internally, but is no longer the public API. 
Update your code to use the new engine interface for forward compatibility.

---

## Configuration

### YAML Config (Unchanged)

```yaml
llm:
  plugin_name: openai
  kwargs:
    model: gpt-4

embedding:
  plugin_name: openai
  kwargs:
    model: text-embedding-3-small

vector_db:
  plugin_name: qdrant
  kwargs:
    url: http://localhost:6333

retriever:
  plugin_name: dense
  collection: my_knowledge
  top_k: 5

rgs:
  max_chunks: 10
  enable_citations: true
  strict_grounding: true
```

Configuration structure remains the same. The new engine wrapper handles loading.

---

## Future Engines

The new architecture enables adding engines without touching core infrastructure:

```python
from fitz.core import Query
from fitz.engines.clara import ClaraEngine  # Future

# Same interface, different paradigm
engine = ClaraEngine(config)
query = Query(text="What is X?")
answer = engine.answer(query)  # CLaRa reasoning instead of RAG
```

All engines implement the same `KnowledgeEngine` protocol:
```python
def answer(self, query: Query) -> Answer:
    ...
```

---

## Examples

See `examples/` directory:
- `quickstart.py` - Basic usage
- `full_pipeline.py` - Complete example with all features
- `ingestion_example.py` - Document ingestion

---

## Contributing

The engine architecture makes it easier to contribute:
- Add new engines in `fitz/engines/<name>/`
- Share infrastructure (`llm/`, `vector_db/`, etc.)
- Don't touch core contracts unless absolutely necessary

---

## Support

- Documentation: https://fitz.readthedocs.io
- Issues: https://github.com/yourorg/fitz/issues
- Discussions: https://github.com/yourorg/fitz/discussions
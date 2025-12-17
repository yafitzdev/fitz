# README.md
# Fitz v0.3.0 - Multi-Engine Knowledge Platform

**Fitz** is a modular knowledge engine platform that supports multiple retrieval-generation paradigms. Build RAG applications today, switch to CLaRa tomorrow, or create your own custom engine.

[![Tests](https://img.shields.io/github/actions/workflow/status/yafitzdev/fitz/ci.yml?label=tests)](https://github.com/yafitzdev/fitz/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## What's New in v0.3.0

- ✅ **Multi-Engine Architecture**: Pluggable engine system with standardized interfaces
- ✅ **CLaRa Engine**: Apple's Continuous Latent Reasoning with 16x-128x document compression
- ✅ **Universal Runtime**: `run(query, engine="clara")` - switch engines with one parameter
- ✅ **Custom Engine Support**: Implement `KnowledgeEngine` protocol and register
- ✅ **Forward Compatible**: Your code won't break when new engines are added

---

## Quick Start

### Installation

```bash
pip install fitz

# With CLaRa support (requires transformers + torch)
pip install fitz[clara]

# With local development support
pip install fitz[local]
```

### Basic Usage

```python
from fitz import run

# Use Classic RAG (default)
answer = run("What is quantum computing?")
print(answer.text)

# Use CLaRa engine
answer = run("Explain quantum entanglement", engine="clara")
print(answer.text)

# View sources
for source in answer.provenance:
    print(f"  - {source.source_id}: {source.excerpt[:100]}...")
```

### Engine-Specific Usage

```python
# Classic RAG
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is machine learning?")

# CLaRa (Continuous Latent Reasoning)
from fitz.engines.clara import run_clara

answer = run_clara(
    "What is quantum computing?",
    documents=["Quantum computers use qubits...", "Unlike classical bits..."]
)
```

---

## Available Engines

| Engine | Description | Best For |
|--------|-------------|----------|
| `classic_rag` | Traditional Retrieval-Augmented Generation | General knowledge bases, production deployments |
| `clara` | Apple's CLaRa with document compression | Large document sets, multi-hop reasoning |

### Classic RAG

Traditional retrieve-then-generate pipeline:
1. Embed query → Vector search → Retrieve chunks
2. Rerank (optional) → Build context
3. Generate answer with LLM

```python
from fitz.engines.classic_rag import run_classic_rag, create_classic_rag_engine

# Quick query
answer = run_classic_rag("What is X?")

# Reusable engine
engine = create_classic_rag_engine("config.yaml")
answer = engine.answer(Query(text="What is Y?"))
```

### CLaRa (Continuous Latent Reasoning)

Apple's compression-native RAG paradigm:
1. Compress documents into continuous memory tokens (16x-128x)
2. Query in latent space via cosine similarity
3. Generate from compressed representations

```python
from fitz.engines.clara import run_clara, create_clara_engine

# Quick query with documents
answer = run_clara(
    "What causes climate change?",
    documents=climate_docs
)

# Reusable engine
engine = create_clara_engine()
engine.add_documents(my_documents)
answer = engine.answer(Query(text="Summarize the key findings"))
```

**CLaRa Advantages:**
- 16x-128x document compression while preserving semantics
- No separate embedding model needed
- Superior multi-hop reasoning
- Unified retrieval-generation optimization

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    User Application                     │
│   answer = run("Question?", engine="clara")             │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Universal Runtime                      │
│   - Engine registry & discovery                         │
│   - Config loading & validation                         │
│   - Query → Answer standardization                      │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ ClassicRAG │  │   CLaRa    │  │  Custom    │
   │  Engine    │  │   Engine   │  │  Engine    │
   └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              ┌────────────────────┐
              │  Shared Services   │
              │  - LLM providers   │
              │  - Vector DBs      │
              │  - Ingestion       │
              └────────────────────┘
```

### Directory Structure

```
fitz/
├── core/                      # Paradigm-agnostic contracts
│   ├── engine.py              # KnowledgeEngine protocol
│   ├── query.py               # Query type
│   ├── answer.py              # Answer type
│   ├── provenance.py          # Source attribution
│   └── exceptions.py          # Error hierarchy
├── engines/
│   ├── classic_rag/           # Traditional RAG
│   │   ├── engine.py          # ClassicRagEngine
│   │   ├── runtime.py         # run_classic_rag()
│   │   ├── pipeline/          # RAG pipeline
│   │   ├── retrieval/         # Retrieval plugins
│   │   └── generation/        # Generation logic
│   └── clara/                 # CLaRa engine
│       ├── engine.py          # ClaraEngine
│       ├── runtime.py         # run_clara()
│       └── config/            # CLaRa configuration
├── runtime/                   # Multi-engine orchestration
│   ├── registry.py            # Engine registry
│   └── runner.py              # Universal run()
├── llm/                       # Shared LLM service
├── vector_db/                 # Shared vector DB service
└── ingest/                    # Document ingestion
```

---

## Custom Engines

Create your own engine by implementing the `KnowledgeEngine` protocol:

### 1. Implement the Protocol

```python
from fitz.core import Query, Answer, Provenance

class MyCustomEngine:
    """Custom knowledge engine."""
    
    def __init__(self, config=None):
        self.config = config
        # Your initialization...
    
    def answer(self, query: Query) -> Answer:
        """Execute query and return answer."""
        # Your logic here...
        
        return Answer(
            text="Generated answer based on my custom logic",
            provenance=[
                Provenance(
                    source_id="source_1",
                    excerpt="Relevant excerpt...",
                    metadata={"relevance_score": 0.95}
                )
            ],
            metadata={"engine": "my_custom", "model": "custom-v1"}
        )
```

### 2. Register with the Registry

```python
from fitz.runtime import EngineRegistry

# Option A: Direct registration
registry = EngineRegistry.get_global()
registry.register(
    name="my_custom",
    factory=lambda config: MyCustomEngine(config),
    description="My custom knowledge engine"
)

# Option B: Decorator
@EngineRegistry.register_engine(name="my_custom", description="My custom engine")
def create_my_engine(config):
    return MyCustomEngine(config)
```

### 3. Use Your Engine

```python
from fitz import run

# Via universal runtime
answer = run("What is X?", engine="my_custom")

# List all available engines
from fitz.runtime import list_engines
print(list_engines())  # ['classic_rag', 'clara', 'my_custom']
```

---

## Configuration

### YAML Configuration

```yaml
# config.yaml

# LLM provider
llm:
  plugin_name: openai
  kwargs:
    model: gpt-4
    temperature: 0.1

# Embedding provider
embedding:
  plugin_name: openai
  kwargs:
    model: text-embedding-3-small

# Vector database
vector_db:
  plugin_name: qdrant
  kwargs:
    url: http://localhost:6333

# Retriever settings
retriever:
  plugin_name: dense
  collection: my_knowledge
  top_k: 5

# Generation settings
rgs:
  max_chunks: 10
  enable_citations: true
  strict_grounding: true
```

### CLaRa Configuration

```yaml
# clara_config.yaml

clara:
  model:
    model_name_or_path: "apple/CLaRa-7B-E2E"
    variant: "e2e"
    device: "cuda"
    torch_dtype: "bfloat16"
  
  compression:
    compression_rate: 16  # 16x, 32x, 64x, or 128x
    doc_max_length: 2048
  
  retrieval:
    top_k: 5
    differentiable_topk: true
  
  generation:
    max_new_tokens: 256
    temperature: 0.7
```

### Local Development (No API Keys)

```yaml
# local_config.yaml

llm:
  plugin_name: local
  kwargs:
    model: llama3.2

embedding:
  plugin_name: local
  kwargs:
    model: nomic-embed-text

vector_db:
  plugin_name: local-faiss
  kwargs:
    index_path: ./data/faiss_index
```

```bash
# Start Ollama
ollama pull llama3.2
ollama pull nomic-embed-text

# Run with local config
fitz-pipeline query "What is X?" --config local_config.yaml
```

---

## CLI

### Query Command

```bash
# Default engine (classic_rag)
fitz query "What is quantum computing?"

# Specific engine
fitz query "Explain X" --engine clara

# With constraints
fitz query "What is Y?" --max-sources 5

# With custom config
fitz query "Analyze this" --config my_config.yaml
```

### Engine Management

```bash
# List available engines
fitz engines

# List with descriptions
fitz engines --verbose
```

### Ingestion

```bash
# Ingest documents
fitz-ingest run ./documents --collection my_kb

# Query the collection
fitz query "What is in my documents?" --collection my_kb
```

---

## Migration from v0.2.x

### Breaking Changes

| v0.2.x | v0.3.0 |
|--------|--------|
| `from fitz.pipeline import RAGPipeline` | `from fitz.engines.classic_rag import run_classic_rag` |
| `pipeline.run("query")` | `run_classic_rag("query")` |
| `result.answer` | `answer.text` |
| `result.sources` | `answer.provenance` |
| `source.chunk_id` | `provenance.source_id` |

### Quick Migration

```python
# OLD (v0.2.x)
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.pipeline.config.loader import load_config

config = load_config()
pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")
print(result.answer)

# NEW (v0.3.0)
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is X?")
print(answer.text)
```

See [MIGRATION.md](docs/MIGRATION.md) for detailed upgrade instructions.

---

## Examples

See the `examples/` directory:

- `quickstart.py` - Basic usage with both engines
- `full_pipeline.py` - Complete example with all features
- `custom_engine.py` - Creating a custom engine
- `ingestion_example.py` - Document ingestion

---

## Contributing

The engine architecture makes it easy to contribute:

1. **Add new engines** in `fitz/engines/<name>/`
2. **Share infrastructure** (`llm/`, `vector_db/`, etc.)
3. **Don't modify core contracts** unless absolutely necessary

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- **Documentation**: https://fitz.readthedocs.io
- **Issues**: https://github.com/yafitzdev/fitz/issues
- **Discussions**: https://github.com/yafitzdev/fitz/discussions
- **CLaRa Paper**: https://arxiv.org/abs/2511.18659
# Fitz Engine Architecture

This document explains Fitz's multi-engine architecture and how different knowledge paradigms are supported.

---

## Overview

Fitz v0.3.0 introduces a **pluggable engine architecture** that supports multiple retrieval-generation paradigms through a unified interface.

```
┌─────────────────────────────────────────────────────────┐
│                     User Code                           │
│                                                         │
│   answer = run("Question?", engine="clara")             │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                  Universal Runtime                      │
│                                                         │
│   • Engine registry & discovery                         │
│   • Config loading & validation                         │
│   • Query → Answer standardization                      │
│                                                         │
└─────────────────────────┬───────────────────────────────┘
                          │
          ┌───────────────┼───────────────┬───────────────┐
          ▼               ▼               ▼               ▼
   ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
   │ ClassicRAG │  │   CLaRa    │  │  GraphRAG  │  │  Custom    │
   │  Engine    │  │   Engine   │  │  (future)  │  │  Engine    │
   └────────────┘  └────────────┘  └────────────┘  └────────────┘
```

---

## Core Contracts

All engines implement the same interface, defined in `fitz/core/`:

### KnowledgeEngine Protocol

```python
from typing import Protocol
from fitz.core import Query, Answer

class KnowledgeEngine(Protocol):
    """Protocol that all knowledge engines must implement."""
    
    def answer(self, query: Query) -> Answer:
        """Execute a query and return an answer."""
        ...
```

### Query

```python
@dataclass
class Query:
    text: str                           # The question
    constraints: Optional[Constraints]  # Query-time limits
    metadata: Dict[str, Any]            # Engine hints
```

### Answer

```python
@dataclass
class Answer:
    text: str                      # Generated answer
    provenance: List[Provenance]   # Source attribution
    metadata: Dict[str, Any]       # Engine metadata
```

### Provenance

```python
@dataclass
class Provenance:
    source_id: str              # Unique source identifier
    excerpt: Optional[str]      # Relevant excerpt
    metadata: Dict[str, Any]    # Source metadata
```

---

## Built-in Engines

### Classic RAG

**Location**: `fitz/engines/classic_rag/`

Traditional Retrieval-Augmented Generation:

```
Query → Embed → Vector Search → Rerank → Context Build → LLM → Answer
```

**Characteristics**:
- Separate embedding model for retrieval
- Vector database for storage
- Chunk-based retrieval
- Optional reranking step
- LLM generates from retrieved context

**Best For**:
- General knowledge bases
- Production deployments
- When you need fine-grained control over retrieval

**Usage**:
```python
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is quantum computing?")
```

**Configuration**:
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
```

---

### CLaRa

**Location**: `fitz/engines/clara/`

Apple's Continuous Latent Reasoning:

```
Documents → Compress (16x-128x) → Latent Space
Query → Encode → Cosine Similarity → Top-K → Generate
```

**Characteristics**:
- Documents compressed into continuous memory tokens
- No separate embedding model needed
- Retrieval and generation in same latent space
- End-to-end optimization possible
- 16x-128x compression while preserving semantics

**Best For**:
- Large document collections
- Multi-hop reasoning queries
- Memory-constrained environments
- When unified retrieval-generation is needed

**Usage**:
```python
from fitz.engines.clara import run_clara, create_clara_engine

# Quick query
answer = run_clara("What causes climate change?", documents=docs)

# Reusable engine
engine = create_clara_engine()
engine.add_documents(my_documents)
answer = engine.answer(Query(text="Summarize key findings"))
```

**Configuration**:
```yaml
clara:
  model:
    model_name_or_path: "apple/CLaRa-7B-E2E"
    variant: "e2e"
    device: "cuda"
    
  compression:
    compression_rate: 16
    doc_max_length: 2048
    
  retrieval:
    top_k: 5
    
  generation:
    max_new_tokens: 256
```

**Model Variants**:
| Variant | Model | Use Case |
|---------|-------|----------|
| `base` | `apple/CLaRa-7B-Base` | Document compression |
| `instruct` | `apple/CLaRa-7B-Instruct` | Instruction-following |
| `e2e` | `apple/CLaRa-7B-E2E` | Full retrieval + generation |

---

## Engine Comparison

| Feature | Classic RAG | CLaRa |
|---------|-------------|-------|
| **Document Storage** | Vector embeddings | Compressed memory tokens |
| **Compression** | None (full text) | 16x-128x |
| **Retrieval Model** | Separate embedding model | Built into LLM |
| **Training** | Separate retriever/generator | End-to-end unified |
| **Context Length** | Long (full documents) | Short (compressed) |
| **Multi-hop Reasoning** | Limited | Superior |
| **Setup Complexity** | Higher (multiple models) | Lower (single model) |
| **Production Ready** | Yes | Experimental |

---

## Engine Registry

The engine registry manages engine discovery and instantiation.

### Listing Engines

```python
from fitz.runtime import list_engines, list_engines_with_info

# Simple list
engines = list_engines()
# ['classic_rag', 'clara']

# With descriptions
info = list_engines_with_info()
# {
#     'classic_rag': 'Traditional RAG with vector retrieval',
#     'clara': 'CLaRa: Compression-native RAG with 16x-128x compression'
# }
```

### Using Engines

```python
from fitz import run
from fitz.runtime import create_engine

# Via universal runtime
answer = run("Question?", engine="clara")

# Create engine instance
engine = create_engine(engine="clara", config=my_config)
answer = engine.answer(Query(text="Question?"))
```

### Engine Registration

Engines auto-register when their module is imported:

```python
# In fitz/engines/clara/__init__.py

def _register_clara_engine():
    from fitz.runtime import EngineRegistry
    
    registry = EngineRegistry.get_global()
    registry.register(
        name="clara",
        factory=lambda config: ClaraEngine(config or ClaraConfig()),
        description="CLaRa: Compression-native RAG"
    )

_register_clara_engine()
```

---

## Choosing an Engine

### Use Classic RAG When:
- ✅ You have a production deployment
- ✅ You need fine-grained retrieval control
- ✅ Your documents are moderate size
- ✅ You want to use existing vector databases
- ✅ You need proven, stable technology

### Use CLaRa When:
- ✅ You have large document collections
- ✅ Queries require multi-hop reasoning
- ✅ Memory/context is constrained
- ✅ You want unified retrieval-generation
- ✅ You're experimenting with new approaches

### Decision Flowchart

```
                    ┌─────────────────┐
                    │  Start Query    │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Production Use? │
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │ Yes                         │ No
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Classic RAG    │           │ Large Doc Set?  │
    └─────────────────┘           └────────┬────────┘
                                           │
                                ┌──────────┼──────────┐
                                │ Yes                 │ No
                                ▼                     ▼
                      ┌─────────────────┐   ┌─────────────────┐
                      │     CLaRa       │   │  Classic RAG    │
                      └─────────────────┘   └─────────────────┘
```

---

## Engine Lifecycle

### Initialization

```python
from fitz.engines.clara import create_clara_engine

# Engine is initialized with config
engine = create_clara_engine(config=my_config)
# → Loads model
# → Sets up internal state
```

### Adding Knowledge

```python
# Add documents to engine
engine.add_documents([
    "Document 1 content...",
    "Document 2 content...",
])
# → Documents are processed (embedded or compressed)
# → Stored in engine's knowledge base
```

### Querying

```python
from fitz.core import Query, Constraints

# Create query with constraints
query = Query(
    text="What is X?",
    constraints=Constraints(max_sources=5)
)

# Execute query
answer = engine.answer(query)
# → Retrieves relevant knowledge
# → Generates answer
# → Returns with provenance
```

### Cleanup

```python
# Clear knowledge base
engine.clear_knowledge_base()

# For cached engines (runtime functions)
from fitz.engines.clara import clear_engine_cache
clear_engine_cache()
```

---

## Future Engines

The architecture is designed to support additional paradigms:

### GraphRAG (Planned v0.3.1)
Knowledge graph-based retrieval:
```
Documents → Extract Entities → Build Graph
Query → Graph Traversal → Subgraph → Generate
```

### Ensemble Engine (Planned v0.4.0)
Combine multiple engines:
```python
answer = run("Question?", engine="ensemble", 
             engines=["classic_rag", "clara"])
```

### Custom Engines
See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md) for creating your own engine.

---

## Architecture Principles

1. **Protocol-Based**: Engines implement protocols, not inherit from base classes
2. **Config-Driven**: Engine behavior controlled by configuration
3. **Isolation**: Engine-specific code stays in engine directory
4. **Shared Infrastructure**: LLM, vector DB, ingestion shared across engines
5. **Standardized I/O**: All engines use Query → Answer
6. **Self-Registration**: Engines register themselves on import

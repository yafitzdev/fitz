# Fitz RAG Engine

This document explains Fitz's RAG engine architecture and core contracts.

---

## Overview

Fitz RAG is a traditional Retrieval-Augmented Generation engine with epistemic guardrails and hierarchical summarization capabilities.

```
Query → Embed → Vector Search → Rerank → Context Build → LLM → Answer
```

**Key Features**:
- Separate embedding model for retrieval
- Vector database for storage (Qdrant, FAISS)
- Chunk-based retrieval with optional reranking
- Hierarchical summaries for analytical queries
- Epistemic guardrails (knows when to say "I don't know")

---

## Core Contracts

All engines implement the same interface, defined in `fitz_ai/core/`:

### KnowledgeEngine Protocol

```python
from typing import Protocol
from fitz_ai.core import Query, Answer

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

## Fitz RAG Engine

**Location**: `fitz_ai/engines/fitz_rag/`

### Usage

```python
from fitz_ai.engines.fitz_rag import run_fitz_rag

answer = run_fitz_rag("What is quantum computing?")
print(answer.text)
print(answer.provenance)
```

### Configuration

```yaml
# fitz.yaml
chat:
  plugin_name: cohere
  kwargs:
    model: command-a-03-2025

embedding:
  plugin_name: cohere
  kwargs:
    model: embed-english-v3.0

vector_db:
  plugin_name: qdrant
  kwargs:
    host: "localhost"
    port: 6333

retriever:
  plugin_name: dense
  collection: my_knowledge
  top_k: 5

rerank:
  enabled: true
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5
```

### Features

| Feature | Description |
|---------|-------------|
| **Hierarchical Summaries** | L0 chunks + L1 doc summaries + L2 corpus summary |
| **Epistemic Guardrails** | Detects contradictions, insufficient evidence |
| **Artifact Generation** | Auto-generates architecture docs, data models, etc. |
| **Incremental Ingestion** | Only re-processes changed files |

---

## Alternative Engines

Fitz supports a pluggable engine architecture. You can swap engines via configuration:

```python
from fitz import run

# Default: Fitz RAG
answer = run("What is X?", engine="fitz_rag")

# Alternative: GraphRAG (knowledge graph-based)
answer = run("What is X?", engine="graphrag")

# Alternative: CLaRa (compression-native, requires GPU)
answer = run("What is X?", engine="clara")
```

### Available Engines

| Engine | Description | Status |
|--------|-------------|--------|
| `fitz_rag` | Traditional RAG with epistemic guardrails | Production |
| `graphrag` | Knowledge graph-based retrieval | Available |
| `clara` | Apple's compression-native RAG (GPU required) | Experimental |

---

## Custom Engines

You can create your own engine. See [CUSTOM_ENGINES.md](CUSTOM_ENGINES.md) for details.

```python
from fitz_ai.core import Query, Answer
from fitz_ai.runtime import EngineRegistry

class MyEngine:
    def answer(self, query: Query) -> Answer:
        # Your logic here
        return Answer(text="...", provenance=[])

# Register and use
EngineRegistry.get_global().register("my_engine", lambda c: MyEngine())
```

---

## Architecture Principles

1. **Protocol-Based**: Engines implement protocols, not inherit from base classes
2. **Config-Driven**: Engine behavior controlled by configuration
3. **Shared Infrastructure**: LLM, vector DB, ingestion shared across engines
4. **Standardized I/O**: All engines use Query → Answer

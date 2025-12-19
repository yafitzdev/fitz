# Migration Guide: v0.2.x → v0.3.0

This guide helps you upgrade your Fitz application from v0.2.x to v0.3.0.

---

## Overview

v0.3.0 is a **major architectural change** that transforms Fitz from a RAG framework into a multi-engine knowledge platform. While this introduces breaking changes, the migration is straightforward and the new architecture provides significant benefits.

### Why the Breaking Changes?

1. **Future-proofing**: Support for multiple paradigms (RAG, CLaRa, GraphRAG, custom)
2. **Cleaner contracts**: Standardized `Query → Engine → Answer` flow
3. **Better separation**: Engine-specific code isolated from shared infrastructure
4. **Extensibility**: Add new engines without modifying core code

---

## Quick Migration (5 Minutes)

If you just want to get your code working:

### Step 1: Update Imports

Find and replace these patterns:

| Find | Replace |
|------|---------|
| `from fitz_ai.pipeline.pipeline.engine import RAGPipeline` | `from fitz_ai.engines.classic_rag import run_classic_rag` |
| `from fitz_ai.pipeline.config.loader import load_config` | `from fitz_ai.engines.classic_rag.config import load_config` |
| `from fitz_ai.core.llm` | `from fitz_ai.llm` |
| `from fitz_ai.core.embedding` | `from fitz_ai.llm.embedding` |
| `from fitz_ai.core.vector_db` | `from fitz_ai.vector_db` |

### Step 2: Update Entry Point

```python
# OLD
config = load_config()
pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")
print(result.answer)

# NEW
from fitz_ai.engines.classic_rag import run_classic_rag
answer = run_classic_rag("What is X?")
print(answer.text)
```

### Step 3: Update Answer Access

```python
# OLD
print(result.answer)
for source in result.sources:
    print(source.chunk_id)
    print(source.text)

# NEW
print(answer.text)
for prov in answer.provenance:
    print(prov.source_id)
    print(prov.excerpt)
```

### Step 4: Run Tests

```bash
pytest
```

---

## Detailed Migration

### Import Path Changes

#### Core Types

```python
# OLD
from fitz_ai.pipeline.models.answer import RGSAnswer
from fitz_ai.pipeline.models.source import RGSSourceRef

# NEW
from fitz_ai.core import Answer, Provenance
```

#### LLM Services

```python
# OLD
from fitz_ai.core.llm.chat.engine import ChatEngine
from fitz_ai.core.llm.embedding.engine import EmbeddingEngine
from fitz_ai.core.llm.rerank.engine import RerankEngine

# NEW
from fitz_ai.llm.chat import ChatEngine
from fitz_ai.llm.embedding import EmbeddingEngine
from fitz_ai.llm.rerank import RerankEngine
```

#### Vector Database

```python
# OLD
from fitz_ai.core.vector_db.engine import VectorDBEngine

# NEW
from fitz_ai.vector_db import VectorDBEngine
```

#### Configuration

```python
# OLD
from fitz_ai.pipeline.config.loader import load_config
from fitz_ai.pipeline.config.schema import FitzConfig

# NEW
from fitz_ai.engines.classic_rag.config import load_config
from fitz_ai.engines.classic_rag.config.schema import FitzConfig
# Or use the new ClassicRagConfig alias
from fitz_ai.engines.classic_rag.config import ClassicRagConfig
```

---

### API Changes

#### Running Queries

**Old API (v0.2.x)**:
```python
from fitz_ai.pipeline.pipeline.engine import RAGPipeline
from fitz_ai.pipeline.config.loader import load_config

# Load config and create pipeline
config = load_config("config.yaml")
pipeline = RAGPipeline.from_config(config)

# Run query
result = pipeline.run("What is quantum computing?")

# Access results
print(result.answer)
print(result.sources)
print(result.metadata)
```

**New API (v0.3.0)**:
```python
# Option 1: Simple function (recommended for most cases)
from fitz_ai.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is quantum computing?")
print(answer.text)
print(answer.provenance)
print(answer.metadata)

# Option 2: Universal runtime (for multi-engine apps)
from fitz import run

answer = run("What is quantum computing?", engine="classic_rag")

# Option 3: Reusable engine (for repeated queries)
from fitz_ai.engines.classic_rag import create_classic_rag_engine
from fitz_ai.core import Query

engine = create_classic_rag_engine("config.yaml")
query = Query(text="What is quantum computing?")
answer = engine.answer(query)
```

#### Answer Object Changes

**Old `RGSAnswer`**:
```python
class RGSAnswer:
    answer: str           # The generated text
    sources: List[...]    # Source references
    metadata: Dict        # Additional info
```

**New `Answer`**:
```python
@dataclass
class Answer:
    text: str                      # The generated text (was: answer)
    provenance: List[Provenance]   # Source references (was: sources)
    metadata: Dict[str, Any]       # Additional info
```

#### Source/Provenance Changes

**Old `RGSSourceRef`**:
```python
class RGSSourceRef:
    chunk_id: str
    text: str
    metadata: Dict
```

**New `Provenance`**:
```python
@dataclass
class Provenance:
    source_id: str           # Was: chunk_id
    excerpt: Optional[str]   # Was: text
    metadata: Dict[str, Any]
```

---

### Configuration Changes

YAML configuration files are **unchanged**. The same `config.yaml` works in both versions.

```yaml
# This works in both v0.2.x and v0.3.0
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
```

---

### CLI Changes

CLI commands are largely unchanged, but now support engine selection:

```bash
# v0.2.x
fitz-pipeline query "What is X?"

# v0.3.0 (backwards compatible)
fitz-pipeline query "What is X?"

# v0.3.0 (new: specify engine)
fitz query "What is X?" --engine classic_rag
fitz query "What is X?" --engine clara
```

New commands:
```bash
# List available engines
fitz engines
fitz engines --verbose
```

---

## Code Examples

### Example 1: Basic Query Migration

```python
# ============= OLD (v0.2.x) =============
from fitz_ai.pipeline.pipeline.engine import RAGPipeline
from fitz_ai.pipeline.config.loader import load_config

def answer_question(question: str) -> str:
    config = load_config()
    pipeline = RAGPipeline.from_config(config)
    result = pipeline.run(question)
    return result.answer

# ============= NEW (v0.3.0) =============
from fitz_ai.engines.classic_rag import run_classic_rag

def answer_question(question: str) -> str:
    answer = run_classic_rag(question)
    return answer.text
```

### Example 2: With Sources

```python
# ============= OLD (v0.2.x) =============
def get_answer_with_sources(question):
    result = pipeline.run(question)
    sources = []
    for src in result.sources:
        sources.append({
            "id": src.chunk_id,
            "text": src.text,
            "metadata": src.metadata
        })
    return {"answer": result.answer, "sources": sources}

# ============= NEW (v0.3.0) =============
def get_answer_with_sources(question):
    answer = run_classic_rag(question)
    sources = []
    for prov in answer.provenance:
        sources.append({
            "id": prov.source_id,
            "text": prov.excerpt,
            "metadata": prov.metadata
        })
    return {"answer": answer.text, "sources": sources}
```

### Example 3: Reusable Engine

```python
# ============= OLD (v0.2.x) =============
class RAGService:
    def __init__(self, config_path: str):
        config = load_config(config_path)
        self.pipeline = RAGPipeline.from_config(config)
    
    def query(self, question: str):
        return self.pipeline.run(question)

# ============= NEW (v0.3.0) =============
from fitz_ai.engines.classic_rag import create_classic_rag_engine
from fitz_ai.core import Query

class RAGService:
    def __init__(self, config_path: str):
        self.engine = create_classic_rag_engine(config_path)
    
    def query(self, question: str):
        return self.engine.answer(Query(text=question))
```

### Example 4: Multi-Engine Application

```python
# ============= NEW (v0.3.0 only) =============
from fitz import run
from fitz_ai.runtime import list_engines

class KnowledgeService:
    def query(self, question: str, engine: str = "classic_rag"):
        return run(question, engine=engine)
    
    def available_engines(self):
        return list_engines()

# Usage
service = KnowledgeService()

# Use Classic RAG for general queries
answer = service.query("What is machine learning?", engine="classic_rag")

# Use CLaRa for complex multi-hop queries
answer = service.query("How do X, Y, and Z relate?", engine="clara")
```

---

## Troubleshooting

### ImportError: No module named 'fitz.pipeline'

The `fitz.pipeline` module has been moved. Update your imports:

```python
# OLD
from fitz_ai.pipeline.pipeline.engine import RAGPipeline

# NEW
from fitz_ai.engines.classic_rag import run_classic_rag
```

### AttributeError: 'Answer' object has no attribute 'answer'

The attribute was renamed:

```python
# OLD
print(result.answer)

# NEW
print(answer.text)
```

### AttributeError: 'Answer' object has no attribute 'sources'

The attribute was renamed:

```python
# OLD
for source in result.sources:
    print(source.chunk_id)

# NEW
for prov in answer.provenance:
    print(prov.source_id)
```

### ModuleNotFoundError: No module named 'transformers'

CLaRa requires additional dependencies:

```bash
pip install fitz[clara]
# or
pip install transformers torch
```

---

## Testing Your Migration

After migrating, run these checks:

```bash
# 1. Run your tests
pytest

# 2. Check architecture compliance
python -m tools.contract_map --fail-on-errors

# 3. Verify imports
python -c "from fitz import run; print('✓ Universal runtime')"
python -c "from fitz_ai.engines.classic_rag import run_classic_rag; print('✓ Classic RAG')"
python -c "from fitz_ai.core import Query, Answer, Provenance; print('✓ Core types')"
```

---

## Getting Help

- **Issues**: https://github.com/yafitzdev/fitz/issues
- **Discussions**: https://github.com/yafitzdev/fitz/discussions
- **Documentation**: https://fitz.readthedocs.io

If you encounter migration issues not covered here, please open an issue!

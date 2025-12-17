# fitz

> A modular RAG framework with clean architecture and straightforward configuration.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Version](https://img.shields.io/badge/version-0.2.0-green.svg)](https://github.com/yafitzdev/fitz/releases)

---

## Why fitz?

Most RAG frameworks hide what's happening behind layers of abstraction. **fitz** gives you direct control:

- **See what's happening** — No magic chains, explicit pipeline steps
- **Swap components freely** — Protocol-based plugins, not vendor lock-in  
- **Run locally** — Zero API costs with Ollama + FAISS
- **Configure, don't code** — YAML-driven, sensible defaults

---

## Quick Start

### Local Development (No API Keys)

```bash
# Install
pip install fitz[local]

# Start Ollama (get it from ollama.ai)
ollama pull llama3.2
ollama pull nomic-embed-text

# Ingest documents
fitz-ingest run ./docs --collection my_docs

# Query
fitz-pipeline query "What are the main concepts?" --collection my_docs
```

### With Cloud APIs

```bash
# Install
pip install fitz

# Set your API key
export OPENAI_API_KEY="sk-..."

# Ingest documents
fitz-ingest run ./docs --collection my_docs

# Query
fitz-pipeline query "What are the main concepts?" --collection my_docs
```

---

## Features

### Local-First
Run everything on your machine. No API keys, no rate limits, no costs.

```bash
pip install fitz[local]
# Uses Ollama for LLMs, FAISS for vector search
```

### Flexible Plugins
Swap any component via config:

**LLMs**: OpenAI, Anthropic, Cohere, Azure, Ollama  
**Vector DBs**: Qdrant, FAISS  
**Everything else**: Drop in your own plugins

### Straightforward Config

```yaml
llm:
  plugin_name: openai
  kwargs:
    model: gpt-4
    temperature: 0

retriever:
  collection: my_docs
  top_k: 5

rgs:
  enable_citations: true
  strict_grounding: true
```

---

## Installation

### From Source

```bash
git clone https://github.com/yafitzdev/fitz.git
cd fitz
pip install -e .
```

### Optional Dependencies

```bash
pip install fitz[ingest]  # PDF, DOCX support
pip install fitz[local]   # Ollama + FAISS
```

---

## Usage

### Python API

```python
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.pipeline.config.loader import load_rag_config

# Load config
config = load_rag_config()

# Create pipeline
pipeline = RAGPipeline.from_config(config)

# Query
result = pipeline.run("What is retrieval-guided synthesis?")

print(result.answer)
for source in result.sources:
    print(f"  - {source.doc_id}: {source.content[:100]}")
```

### CLI

```bash
# Ingest documents
fitz-ingest run ./my-documents \
  --collection docs \
  --chunk-size 1000 \
  --chunk-overlap 200

# Query
fitz-pipeline query "Your question here" \
  --collection docs \
  --preset standard

# Show config
fitz-pipeline config show
```

---

## Configuration

### Environment Variables

```bash
# LLM APIs
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
export COHERE_API_KEY="..."

# Vector DB (optional)
export QDRANT_URL="http://localhost:6333"
```

### YAML Config

Place a `config.yaml` in your project:

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

---

## Architecture

```
CLI (fitz, fitz-ingest, fitz-pipeline)
    ↓
Pipeline (orchestration)
    ↓
Plugins (chat, embed, rerank, vector_db)
    ↓
Core (registry, config, models)
    ↓
Backends (ollama, faiss, cloud APIs)
```

Clean boundaries. No circular dependencies.

---

## Examples

### Local RAG Pipeline

```python
from fitz.pipeline.pipeline.engine import RAGPipeline
from fitz.pipeline.config.schema import RAGConfig, PipelinePluginConfig

# Local config
config = RAGConfig(
    llm=PipelinePluginConfig(
        plugin_name="local",
        kwargs={"model": "llama3.2"}
    ),
    embedding=PipelinePluginConfig(
        plugin_name="local",
        kwargs={"model": "nomic-embed-text"}
    ),
    vector_db=PipelinePluginConfig(
        plugin_name="local-faiss",
        kwargs={}
    ),
    retriever=RetrieverConfig(
        plugin_name="dense",
        collection="docs",
        top_k=5
    )
)

pipeline = RAGPipeline.from_config(config)
result = pipeline.run("What is X?")
```

### Custom Ingestion

```python
from fitz.ingest.pipeline.ingestion_pipeline import IngestionPipeline
from fitz.ingest.config.schema import IngestConfig

config = IngestConfig(
    ingester=IngesterConfig(
        plugin_name="local",
        kwargs={}
    ),
    chunker=ChunkerConfig(
        plugin_name="simple",
        chunk_size=1000,
        chunk_overlap=200,
        kwargs={}
    ),
    collection="my_docs"
)

pipeline = IngestionPipeline(config)
pipeline.run(source="./documents")
```

---

## Plugin Development

### Create a Chat Plugin

```python
from fitz.llm.chat.base import ChatPlugin

class MyChatClient(ChatPlugin):
    plugin_name = "my_llm"
    
    def __init__(self, model: str = "default", **kwargs):
        self.model = model
    
    def chat(self, messages: list[dict[str, Any]]) -> str:
        # Your implementation
        return response_text
```

Place it in `fitz/core/llm/chat/plugins/` and it's automatically discovered.

### Use It

```yaml
llm:
  plugin_name: my_llm
  kwargs:
    model: custom-model
```

---

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fitz

# Run specific test
pytest tests/test_rag_pipeline_end_to_end.py
```

---

## Development

```bash
# Clone
git clone https://github.com/yafitzdev/fitz.git
cd fitz

# Install for development
pip install -e ".[local,ingest]"

# Run tests
pytest

# Format code
black .
isort .
```

---

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas we'd love help with:
- Additional LLM/vector DB plugins
- Advanced chunking strategies
- Performance improvements
- Documentation

---

## License

MIT License - see [LICENSE](LICENSE) for details.

---

## Links

- **Repository**: https://github.com/yafitzdev/fitz
- **Issues**: https://github.com/yafitzdev/fitz/issues
- **Changelog**: [CHANGELOG.md](CHANGELOG.md)

---

**Build RAG pipelines that make sense.**
# fitz-ai

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyPI version](https://badge.fury.io/py/fitz-ai.svg)](https://pypi.org/project/fitz-ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.5-green.svg)](CHANGELOG.md)

## 🎯 Stable Knowledge Access, Today and Tomorrow

fitz-ai is a **knowledge access platform** for teams that need reliable, configurable retrieval **today**, without locking themselves into a single reasoning paradigm **tomorrow**.

You ingest your knowledge once. How it gets queried can evolve.

---

## 🤔 Why fitz-ai Exists

Organizations repeatedly rebuild the same systems: ingest documents, chunk them, embed them, retrieve them, generate answers. Every time the reasoning method changes, everything breaks.

**The insight:** Reasoning methods evolve faster than knowledge.

- RAG today
- Compression-native models tomorrow
- Something else after that

But the knowledge layer remains.

Most RAG tools optimize *one method*. fitz-ai stabilizes the **knowledge layer itself**.

---

## 🧠 The Mental Model

```
  Your Knowledge
      ↓
  fitz-ai (Knowledge Access Layer)
      ↓
  Engines (replaceable)
      ↓
  Answer
```

**What stays stable:** Ingested documents, chunking decisions, metadata, provenance, API contracts.

**What can change:** Retrieval strategies, reasoning methods, model providers, compression techniques.

You optimize for **stability where it matters** and **flexibility where change is inevitable**.

---

## ⚖️ How fitz-ai Is Different

This isn't a critique of other tools. It's a design difference.

| | LangChain & Similar | fitz-ai |
|---|---------------------|------|
| **Optimizes for** | Flows & prompt chains | Knowledge stability |
| **Assumes** | Rapid experimentation | Systems live for years |
| **Switching paradigms** | Often means refactoring | Means changing engines |
| **Best for** | Exploring ideas | Building infrastructure |

If you're exploring ideas, LangChain is excellent. If you're building infrastructure that will outlive your current model choices, fitz-ai is designed for that.

---

## 🚀 Quick Start

```bash
pip install fitz-ai
```

```python
from fitz_ai.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What does our contract say about termination?")
print(answer.text)
```

That's it. Classic RAG works out of the box.

---

## ⚙️ Engines

Engines encapsulate *how* knowledge is queried. They're not plugins. They're paradigms.

### Classic RAG (Default) ✅

Production-ready retrieval-augmented generation.

```python
from fitz_ai.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is our refund policy?")

for source in answer.provenance:
    print(f"{source.source_id}: {source.excerpt}")
```

### CLaRa (Experimental) 🧪

Compression-native reasoning for large document collections. 16x to 128x compression with unified retrieval and generation.

```python
from fitz_ai.engines.clara import create_clara_engine

engine = create_clara_engine()
engine.add_documents(my_documents)
answer = engine.answer(Query(text="What patterns emerge across these reports?"))
```

> Engines are interchangeable. Your knowledge is not.

---

## ✅ When fitz-ai Makes Sense

- Internal company knowledge bases
- Compliance-sensitive environments
- Teams running local and cloud LLMs
- Long-lived systems where methods will change

## ❌ When fitz-ai Is Not a Fit

- Prompt-only experiments
- One-off demos
- No ingestion, no retrieval needed

---

## 📁 Project Structure

```
fitz_ai/
├── core/        # Stable contracts (Query, Answer, Provenance)
├── engines/     # Reasoning paradigms (classic_rag, clara)
├── ingest/      # Knowledge ingestion
├── runtime/     # Engine orchestration
├── llm/         # LLM plugins
└── vector_db/   # Vector DB plugins
```

Architecture enforces separation: engines can be added or removed without destabilizing the core.

---

## 💻 CLI

### Core Commands

| Command | Description |
|---------|-------------|
| `fitz init` | Interactive setup wizard |
| `fitz ingest [path]` | Ingest documents into vector DB |
| `fitz query "question"` | Query your knowledge base |
| `fitz config` | View/manage configuration |
| `fitz doctor` | System diagnostics |

### Quick Start

```bash
# 1. Setup (detects Ollama, Qdrant, API keys automatically)
fitz init

# 2. Ingest documents
fitz ingest ./my-documents

# 3. Query
fitz query "What are the main topics?"

# 4. Verify everything works
fitz doctor --test
```

### Command Details

#### `fitz init`
Interactive setup wizard. Detects available providers and creates config.

```bash
fitz init              # Interactive mode
fitz init -y           # Auto-detect defaults
fitz init --show       # Preview without saving
```

#### `fitz ingest`
Ingest documents into your vector database.

```bash
fitz ingest                    # Interactive prompts
fitz ingest ./docs             # Ingest specific directory
fitz ingest ./docs -y          # Non-interactive with defaults
```

Prompts for: collection name, chunker type, chunk size, overlap.

#### `fitz query`
Query your knowledge base with RAG.

```bash
fitz query "your question"              # Basic query
fitz query "your question" --stream     # Streaming response
```

Returns answer with source citations.

#### `fitz config`
Manage configuration.

```bash
fitz config              # Show summary
fitz config --raw        # Show YAML
fitz config --json       # JSON output
fitz config --edit       # Open in $EDITOR
fitz config --path       # Show file location
```

#### `fitz doctor`
System diagnostics.

```bash
fitz doctor              # Quick check
fitz doctor -v           # Verbose (shows optional deps)
fitz doctor --test       # Test connections
```

Checks: Python version, dependencies, Ollama/Qdrant/FAISS availability, API keys, connections.

### Examples

```bash
# Local-first setup (no API keys)
ollama serve
docker run -p 6333:6333 qdrant/qdrant
fitz init  # Select ollama + qdrant
fitz ingest ./docs -y
fitz query "What's in my docs?"

# Cloud setup
export COHERE_API_KEY="your-key"
fitz init -y
fitz ingest ./docs -y
fitz query "Your question"

# Check everything is working
fitz doctor --test
```

### Environment Variables

```bash
# API Keys (choose one or use Ollama)
export COHERE_API_KEY="..."       # Recommended
export OPENAI_API_KEY="..."       # Alternative
export AZURE_OPENAI_API_KEY="..." # Alternative

# Azure specific (if using Azure)
export AZURE_OPENAI_ENDPOINT="..."
export AZURE_OPENAI_API_VERSION="2024-02-15-preview"
```

### Configuration File

Created at `~/.fitz/fitz.yaml`:

```yaml
chat:
  plugin_name: cohere
  kwargs:
    model: command-a-03-2025
    temperature: 0.2

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
  collection: default
  top_k: 5

rerank:
  enabled: true
  plugin_name: cohere
  kwargs:
    model: rerank-v3.5

rgs:
  enable_citations: true
  strict_grounding: true
  max_chunks: 8
```

Edit with: `fitz config --edit` or `vim ~/.fitz/fitz.yaml`

---

## 📐 Design Principles

- **Explicit over clever** | No hidden magic
- **Stable contracts** | The API doesn't break when internals change
- **Knowledge outlives methods** | Ingest once, query many ways
- **Engines are paradigms** | Not just config switches

---

## 💡 Philosophy

RAG is a method.  
Knowledge access is a strategy.

fitz-ai is built for the strategy.

---

## 📚 Documentation

- [Engine Guide](docs/ENGINES.md) | Choosing and using engines
- [Architecture](docs/architecture.md) | Deep dive for contributors
- [Changelog](CHANGELOG.md) | Release history

---

## 📄 License

MIT
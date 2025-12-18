# Fitz 🎯 Stable Knowledge Access, Today and Tomorrow

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.3.1-green.svg)](CHANGELOG.md)

Fitz is a **knowledge access platform** for teams that need reliable, configurable retrieval **today**, without locking themselves into a single reasoning paradigm **tomorrow**.

You ingest your knowledge once. How it gets queried can evolve.

---

## 🤔 Why Fitz Exists

Organizations repeatedly rebuild the same systems: ingest documents, chunk them, embed them, retrieve them, generate answers. Every time the reasoning method changes, everything breaks.

**The insight:** Reasoning methods evolve faster than knowledge.

- RAG today
- Compression-native models tomorrow
- Something else after that

But the knowledge layer remains.

Most RAG tools optimize *one method*. Fitz stabilizes the **knowledge layer itself**.

---

## 🧠 The Mental Model

```
  Your Knowledge
      ↓
  Fitz (Knowledge Access Layer)
      ↓
  Engines (replaceable)
      ↓
  Answer
```

**What stays stable:** Ingested documents, chunking decisions, metadata, provenance, API contracts.

**What can change:** Retrieval strategies, reasoning methods, model providers, compression techniques.

You optimize for **stability where it matters** and **flexibility where change is inevitable**.

---

## ⚖️ How Fitz Is Different

This isn't a critique of other tools. It's a design difference.

| | LangChain & Similar | Fitz |
|---|---------------------|------|
| **Optimizes for** | Flows & prompt chains | Knowledge stability |
| **Assumes** | Rapid experimentation | Systems live for years |
| **Switching paradigms** | Often means refactoring | Means changing engines |
| **Best for** | Exploring ideas | Building infrastructure |

If you're exploring ideas, LangChain is excellent. If you're building infrastructure that will outlive your current model choices, Fitz is designed for that.

---

## 🚀 Quick Start

```bash
pip install fitz
```

```python
from fitz.engines.classic_rag import run_classic_rag

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
from fitz.engines.classic_rag import run_classic_rag

answer = run_classic_rag("What is our refund policy?")

for source in answer.provenance:
    print(f"{source.source_id}: {source.excerpt}")
```

### CLaRa (Experimental) 🧪

Compression-native reasoning for large document collections. 16x to 128x compression with unified retrieval and generation.

```python
from fitz.engines.clara import create_clara_engine

engine = create_clara_engine()
engine.add_documents(my_documents)
answer = engine.answer(Query(text="What patterns emerge across these reports?"))
```

> Engines are interchangeable. Your knowledge is not.

---

## ✅ When Fitz Makes Sense

- Internal company knowledge bases
- Compliance-sensitive environments
- Teams running local and cloud LLMs
- Long-lived systems where methods will change

## ❌ When Fitz Is Not a Fit

- Prompt-only experiments
- One-off demos
- No ingestion, no retrieval needed

---

## 📁 Project Structure

```
fitz/
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
| `fitz query "question"` | Query your knowledge base |
| `fitz config` | Show current configuration |
| `fitz db` | List/inspect vector collections |
| `fitz chunk ./file.txt` | Preview chunking strategies |
| `fitz doctor` | System diagnostics |
| `fitz plugins` | List all available plugins |

### Ingestion Commands

| Command | Description |
|---------|-------------|
| `fitz ingest ./docs collection` | Ingest documents into collection |
| `fitz ingest ./docs coll --chunk-size 500` | Custom chunk size |
| `fitz ingest validate ./docs` | Validate before ingesting |
| `fitz ingest plugins` | List ingest plugins |

### Database Commands

| Command | Description |
|---------|-------------|
| `fitz db` | List all collections |
| `fitz db default` | Inspect 'default' collection |
| `fitz db my_docs -n 10` | Show 10 sample chunks |

### Chunking Preview

| Command | Description |
|---------|-------------|
| `fitz chunk ./doc.txt` | Preview with defaults (1000 chars) |
| `fitz chunk ./doc.txt --size 500` | Smaller chunks |
| `fitz chunk ./docs/ --stats` | Stats only, no content |
| `fitz chunk --list` | List available chunkers |

### Examples

```bash
# Setup and first query
fitz init
fitz ingest ./documents knowledge_base
fitz query "What are the main topics?"

# Inspect what's stored
fitz db knowledge_base

# Preview chunking before committing
fitz chunk ./large_doc.pdf --size 500 --stats

# Check system health
fitz doctor
```

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

Fitz is built for the strategy.

---

## 📚 Documentation

- [Engine Guide](docs/ENGINES.md) | Choosing and using engines
- [Architecture](docs/architecture.md) | Deep dive for contributors  
- [Migration Guide](docs/MIGRATION.md) | Upgrading from previous versions
- [Changelog](CHANGELOG.md) | Release history

---

## 📄 License

MIT
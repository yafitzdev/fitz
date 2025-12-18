# CONTRIBUTING.md
# Contributing to Fitz

Thank you for your interest in contributing to Fitz! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Guidelines](#architecture-guidelines)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
- [Engine Development](#engine-development)
- [Plugin Development](#plugin-development)
- [Testing](#testing)
- [Style Guide](#style-guide)

---

## Code of Conduct

Be respectful, inclusive, and constructive. We're all here to build something useful together.

---

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment
4. Create a branch for your work
5. Make your changes
6. Submit a pull request

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/fitz.git
cd fitz

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev,local,ingest]"

# Verify setup
pytest
python -m tools.contract_map --layout-depth 2
```

---

## Architecture Guidelines

Fitz follows strict architectural principles. Please respect these when contributing.

### Project Structure (v0.3.0+)

```
fitz/
├── core/              # Paradigm-agnostic contracts (Query, Answer, Provenance)
├── engines/           # Engine implementations
│   ├── classic_rag/   # Traditional RAG engine
│   └── clara/         # CLaRa compression-native engine
├── runtime/           # Multi-engine orchestration
├── llm/               # Shared LLM service (chat, embedding, rerank)
├── vector_db/         # Shared vector DB service
├── ingest/            # Document ingestion
├── cli/               # Command-line interface
└── backends/          # Local backends (Ollama, FAISS)
```

### Layer Dependencies

```
core/        ← NO imports from engines/, ingest/, or backends/
engines/     ← May import from core/, llm/, vector_db/
llm/         ← May import from core/
vector_db/   ← May import from core/
ingest/      ← May import from core/
runtime/     ← May import from all (orchestration layer)
cli/         ← May import from all (user-facing layer)
backends/    ← May import from core/
tools/       ← May import from all (development tools)
```

**Violation of layer dependencies will block your PR.**

Run the contract map to check for violations:

```bash
python -m tools.contract_map --fail-on-errors
```

### Core Principle: Knowledge → Engine → Answer

All engines implement the same protocol:

```python
from fitz.core import KnowledgeEngine, Query, Answer

class MyEngine(KnowledgeEngine):
    def answer(self, query: Query) -> Answer:
        # Your implementation
        ...
```

### Config-Driven Design

- Engines are built from config
- Provider selection lives only in config files
- No provider-specific code outside plugins

---

## How to Contribute

### Reporting Bugs

Open an issue with:
- Clear description of the bug
- Steps to reproduce
- Expected vs actual behavior
- Python version and OS
- Relevant config/code snippets

### Suggesting Features

Open an issue with:
- Clear description of the feature
- Use case / motivation
- Proposed API or interface (if applicable)
- Willingness to implement

### Contributing Code

1. **Small PRs are better**: Focused changes are easier to review
2. **One concern per PR**: Don't mix refactoring with features
3. **Tests required**: All new code needs tests
4. **Documentation**: Update relevant docs

---

## Pull Request Process

1. **Create a branch**
   ```bash
   git checkout -b feature/my-feature
   # or
   git checkout -b fix/bug-description
   ```

2. **Make your changes**
   - Follow the style guide
   - Add/update tests
   - Update documentation

3. **Run checks locally**
   ```bash
   # Format code
   black .
   isort .
   
   # Type check
   mypy fitz
   
   # Run tests
   pytest
   
   # Check architecture
   python -m tools.contract_map --fail-on-errors
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat(engines): add hybrid retrieval to classic_rag"
   git commit -m "fix(core): handle empty embedding response"
   git commit -m "docs: update engine development guide"
   ```

5. **Push and create PR**
   ```bash
   git push origin feature/my-feature
   ```

6. **PR Description should include:**
   - What changes were made
   - Why (motivation/context)
   - How to test
   - Breaking changes (if any)

---

## Engine Development

Engines are the core abstraction in Fitz. Each engine is a complete implementation of a knowledge retrieval paradigm.

### Creating a New Engine

1. **Create the engine directory**
   ```
   fitz/engines/my_engine/
   ├── __init__.py      # Public API exports
   ├── engine.py        # KnowledgeEngine implementation
   ├── runtime.py       # Convenience functions (run_my_engine, create_my_engine)
   └── config/
       ├── __init__.py
       ├── schema.py    # Pydantic config models
       └── loader.py    # Config loading logic
   ```

2. **Implement the KnowledgeEngine protocol**
   ```python
   # fitz/engines/my_engine/engine.py
   from fitz.core import KnowledgeEngine, Query, Answer, Provenance
   
   class MyEngine:
       """My custom knowledge engine."""
       
       def __init__(self, config: MyEngineConfig):
           self._config = config
           # Initialize your engine
       
       def answer(self, query: Query) -> Answer:
           # Your implementation
           return Answer(
               text="The answer",
               provenance=[Provenance(source_id="doc1", excerpt="...")],
           )
   ```

3. **Register with the engine registry**
   ```python
   # fitz/engines/my_engine/__init__.py
   from fitz.runtime import EngineRegistry
   
   def _register():
       registry = EngineRegistry.get_global()
       registry.register(
           name="my_engine",
           factory=lambda config: MyEngine(config or MyEngineConfig()),
           description="My custom knowledge engine",
       )
   
   _register()
   ```

4. **Add convenience functions**
   ```python
   # fitz/engines/my_engine/runtime.py
   from fitz.core import Answer
   
   def run_my_engine(query: str, **kwargs) -> Answer:
       """Execute a query with MyEngine."""
       engine = create_my_engine(**kwargs)
       return engine.answer(Query(text=query))
   ```

5. **Add tests**
   ```python
   # tests/engines/test_my_engine.py
   def test_my_engine_answers_query():
       engine = MyEngine(MyEngineConfig())
       answer = engine.answer(Query(text="What is X?"))
       assert answer.text
       assert isinstance(answer.provenance, list)
   ```

---

## Plugin Development

Plugins extend functionality within engines (LLM providers, vector DBs, etc.).

### Creating a New Plugin

1. **Identify the plugin type**: `chat`, `embedding`, `rerank`, `vector_db`, `retrieval`, `chunking`, `ingestion`

2. **Create the plugin file**:
   ```python
   # fitz/llm/chat/plugins/my_provider.py
   
   class MyProviderChatClient:
       plugin_name = "my_provider"
       plugin_type = "chat"
       
       def __init__(self, api_key: str = None, **kwargs):
           self.api_key = api_key
       
       def chat(self, messages: list[dict]) -> str:
           # Implement the protocol method
           return "response"
   ```

3. **The plugin will be auto-discovered**: No registration needed

4. **Add tests**:
   ```python
   # tests/test_my_provider_plugin.py
   
   def test_my_provider_chat_basic():
       plugin = MyProviderChatClient(api_key="test")
       response = plugin.chat([{"role": "user", "content": "hello"}])
       assert isinstance(response, str)
   ```

### Plugin Protocol Reference

| Type | Protocol | Required Method | Return Type |
|------|----------|-----------------|-------------|
| `chat` | `ChatPlugin` | `chat(messages)` | `str` |
| `embedding` | `EmbeddingPlugin` | `embed(text)` | `list[float]` |
| `rerank` | `RerankPlugin` | `rerank(query, chunks)` | `list[Chunk]` |
| `vector_db` | `VectorDBPlugin` | `search(collection, vector, limit)` | `list[SearchResult]` |
| `retrieval` | `RetrievalPlugin` | `retrieve(query)` | `list[Chunk]` |
| `chunking` | `ChunkerPlugin` | `chunk_text(text, meta)` | `list[Chunk]` |
| `ingestion` | `IngestPlugin` | `ingest(source, kwargs)` | `Iterable[RawDocument]` |

---

## Testing

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=fitz

# Specific module
pytest tests/engines/test_classic_rag.py

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

### Writing Tests

- Place tests in `tests/` directory
- Name files `test_<module>_<feature>.py`
- Use descriptive test function names
- Test both success and failure cases
- Mock external services (APIs, databases)

```python
# Good test example
def test_classic_rag_preserves_metadata():
    """Classic RAG should preserve document metadata in provenance."""
    engine = create_classic_rag_engine(config)
    answer = engine.answer(Query(text="test query"))
    
    assert answer.provenance
    assert all(p.metadata for p in answer.provenance)
```

---

## Style Guide

### Python Style

- **Formatter**: Black (line length 100)
- **Import sorting**: isort (black profile)
- **Type hints**: Required for public APIs
- **Docstrings**: Google style for public classes/functions

### Naming Conventions

| Item | Convention | Example |
|------|------------|---------|
| Modules | `snake_case` | `classic_rag.py` |
| Classes | `PascalCase` | `ClassicRagEngine` |
| Functions | `snake_case` | `run_classic_rag()` |
| Constants | `UPPER_SNAKE` | `DEFAULT_TOP_K` |
| Plugin names | `snake_case` | `plugin_name = "my_provider"` |
| Engine names | `snake_case` | `engine="classic_rag"` |

### Code Organization

```python
# File structure
"""Module docstring."""

from __future__ import annotations

# Standard library
import os
from typing import Any

# Third party
from pydantic import BaseModel

# Local imports (absolute)
from fitz.core import Answer, Query
from fitz.logging.logger import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_VALUE = 10


# Classes/Functions
class MyClass:
    ...
```

---

## Architecture Compliance Checklist

Before submitting a PR, verify:

- [ ] No imports from `engines/` in `core/`
- [ ] No imports from `ingest/` in `core/`
- [ ] No imports from `backends/` in `core/`
- [ ] New engines implement `KnowledgeEngine` protocol
- [ ] New plugins follow the Protocol pattern
- [ ] Config-driven design (no hardcoded provider selection)
- [ ] Tests added for new functionality
- [ ] `python -m tools.contract_map --fail-on-errors` passes

---

## Questions?

- Open a GitHub issue for questions
- Tag with `question` label
- Check existing issues first

Thank you for contributing! 🎉
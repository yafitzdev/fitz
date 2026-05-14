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
git clone https://github.com/yafitzdev/fitz-sage.git
cd fitz-sage

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[dev]"

# Verify setup
pytest
python -m tools.contract_map --layout-depth 2
```

---

## Architecture Guidelines

Fitz follows strict architectural principles. Please respect these when contributing.

### Project Structure

```
fitz_sage/
├── core/              # Paradigm-agnostic contracts (Query, Answer, Provenance)
├── engines/           # Engine implementations
│   └── fitz_krag/     # KRAG with retrieval intelligence
├── retrieval/         # SHARED retrieval intelligence (detection, entities, vocabulary, etc.)
├── llm/               # LLM service layer — single OpenAI-compatible HTTP protocol (chat + reranker)
├── storage/           # SQLite connection manager (one .db per collection, FTS5 + bm25)
├── ingestion/         # Document ingestion (parser, chunking, enrichment)
├── tabular/           # CSV/table query with SQL generation (SqliteTableStore)
├── runtime/           # Multi-engine orchestration
├── cli/               # Command-line interface
├── api/               # REST API (FastAPI)
└── sdk/               # Stateful Python interface
```

### Layer Dependencies

```
core/        ← NO imports from engines/, ingestion/
retrieval/   ← May import from core/
llm/         ← May import from core/
storage/     ← May import from core/
ingestion/   ← May import from core/
engines/     ← May import from core/, llm/, storage/, retrieval/
runtime/     ← May import from all (orchestration layer)
cli/         ← May import from all (user-facing layer)
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
from fitz_sage.core import KnowledgeEngine, Query, Answer

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
   mypy fitz_sage

   # Run tests
   pytest

   # Check architecture
   python -m tools.contract_map --fail-on-errors
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat(engines): add hybrid retrieval to fitz_krag"
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
   fitz_sage/engines/my_engine/
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
   # fitz_sage/engines/my_engine/engine.py
   from fitz_sage.core import KnowledgeEngine, Query, Answer, Provenance
   
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
   # fitz_sage/engines/my_engine/__init__.py
   from fitz_sage.runtime import EngineRegistry
   
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
   # fitz_sage/engines/my_engine/runtime.py
   from fitz_sage.core import Answer
   
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

1. **Identify the plugin type**: `llm-chat`, `llm-embedding`, `llm-rerank`, `retrieval`, `constraint`, `reader`, `chunker`

2. **Create the plugin file**:
   ```python
   # fitz_sage/llm/chat/plugins/my_provider.py
   
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
pytest --cov=fitz_sage

# Specific module
pytest tests/engines/test_fitz_krag.py

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
def test_fitz_krag_preserves_metadata():
    """Fitz KRAG should preserve document metadata in provenance."""
    engine = create_fitz_krag_engine(config)
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
| Modules | `snake_case` | `fitz_krag.py` |
| Classes | `PascalCase` | `FitzKragEngine` |
| Functions | `snake_case` | `run_fitz_krag()` |
| Constants | `UPPER_SNAKE` | `DEFAULT_TOP_K` |
| Plugin names | `snake_case` | `plugin_name = "my_provider"` |
| Engine names | `snake_case` | `engine="fitz_krag"` |

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
from fitz_sage.core import Answer, Query
from fitz_sage.logging.logger import get_logger

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
- [ ] No imports from `ingestion/` in `core/`
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
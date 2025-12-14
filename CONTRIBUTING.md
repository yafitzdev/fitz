# Contributing to fitz

Thank you for your interest in contributing to fitz! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Architecture Guidelines](#architecture-guidelines)
- [How to Contribute](#how-to-contribute)
- [Pull Request Process](#pull-request-process)
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
git clone https://github.com/yafitzdev/fitz.git
cd fitz

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[docs,ingest,map,ground]"

# Install development dependencies
pip install pytest pytest-cov black isort mypy

# Verify setup
pytest
python -m tools.contract_map --layout-depth 2
```

---

## Architecture Guidelines

fitz follows strict architectural principles. Please respect these when contributing:

### Layer Dependencies

```
core/     ← NO imports from rag/ or ingest/
rag/      ← May import from core/, NOT from ingest/
ingest/   ← May import from core/, NOT from rag/
tools/    ← May import from all packages
```

**Violation of layer dependencies will block your PR.**

Run the contract map to check for violations:

```bash
python -m tools.contract_map --fail-on-errors
```

### Plugin Architecture

All extensibility happens through plugins:

1. **Plugins implement Protocols** — Not base classes
2. **Plugins are auto-discovered** — No manual registration
3. **Plugins declare `plugin_name` and `plugin_type`** — For registry identification
4. **Config selects plugins** — Never hardcode provider selection in logic

### Config-Driven Design

- Engines are built FROM config (`Engine.from_config(cfg)`)
- Provider selection lives ONLY in config files
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

1. **Small PRs are better** — Focused changes are easier to review
2. **One concern per PR** — Don't mix refactoring with features
3. **Tests required** — All new code needs tests
4. **Documentation** — Update relevant docs

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
   mypy core rag ingest
   
   # Run tests
   pytest
   
   # Check architecture
   python -m tools.contract_map --fail-on-errors
   ```

4. **Commit with clear messages**
   ```bash
   git commit -m "feat(rag): add hybrid retrieval plugin"
   git commit -m "fix(core): handle empty embedding response"
   git commit -m "docs: update plugin development guide"
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

## Plugin Development

### Creating a New Plugin

1. **Identify the plugin type** (`chat`, `embedding`, `rerank`, `vector_db`, `retrieval`, `chunking`, `ingestion`)

2. **Create the plugin file** in the appropriate `plugins/` directory:
   ```python
   # core/llm/chat/plugins/my_provider.py
   
   class MyProviderChatClient:
       plugin_name = "my_provider"  # Unique identifier
       plugin_type = "chat"         # Must match registry type
       
       def __init__(self, api_key: str = None, **kwargs):
           # Initialize your client
           pass
       
       def chat(self, messages: list[dict]) -> str:
           # Implement the protocol method
           return "response"
   ```

3. **The plugin will be auto-discovered** — No registration needed

4. **Add tests**:
   ```python
   # tests/test_my_provider_plugin.py
   
   def test_my_provider_chat_basic():
       plugin = MyProviderChatClient(api_key="test")
       response = plugin.chat([{"role": "user", "content": "hello"}])
       assert isinstance(response, str)
   ```

5. **Document in plugin docstring**:
   ```python
   class MyProviderChatClient:
       """
       Chat plugin for MyProvider API.
       
       Required environment variables:
           MY_PROVIDER_API_KEY: API key for authentication
       
       Config example:
           llm:
             plugin_name: my_provider
             kwargs:
               model: my-model-v1
       """
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
pytest --cov=core --cov=rag --cov=ingest

# Specific module
pytest tests/test_retriever_*.py

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
def test_retriever_preserves_metadata():
    """Retriever should pass through all payload fields to chunk metadata."""
    hits = [Hit(id="h", payload={"doc_id": "doc", "custom_field": 123})]
    retriever = DenseRetrievalPlugin(client=MockClient(hits), ...)
    
    out = retriever.retrieve("query")
    
    assert out[0].metadata["custom_field"] == 123
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
| Modules | `snake_case` | `dense_retrieval.py` |
| Classes | `PascalCase` | `DenseRetrievalPlugin` |
| Functions | `snake_case` | `get_retriever_plugin()` |
| Constants | `UPPER_SNAKE` | `RETRIEVER_REGISTRY` |
| Plugin names | `snake_case` | `plugin_name = "my_provider"` |

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
from core.models.chunk import Chunk
from core.logging.logger import get_logger

logger = get_logger(__name__)

# Constants
DEFAULT_VALUE = 10

# Classes/Functions
class MyClass:
    ...
```

---

## Questions?

- Open a GitHub issue for questions
- Tag with `question` label
- Check existing issues first

Thank you for contributing! 🎉

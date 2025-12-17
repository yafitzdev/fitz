# CHANGELOG.md
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2025-12-17

### 🎉 Overview

Fitz v0.3.0 transforms the project from a RAG framework into a **multi-engine knowledge platform**. This release introduces a pluggable engine architecture, the CLaRa engine for compression-native RAG, and a universal runtime that makes switching between paradigms seamless.

---

### ✨ Highlights

#### Multi-Engine Architecture
- **Universal Runtime**: `run(query, engine="clara")` - switch engines with one parameter
- **Engine Registry**: Discover, register, and manage knowledge engines
- **Protocol-Based Design**: Implement `answer(Query) -> Answer` to create custom engines
- **Shared Infrastructure**: LLM, vector DB, and ingestion services shared across engines

#### CLaRa Engine (NEW)
- **Apple's CLaRa Integration**: Continuous Latent Reasoning for RAG
- **16x-128x Document Compression**: Preserve semantics while drastically reducing context
- **Unified Retrieval-Generation**: Single model for both retrieval and generation
- **Multi-Hop Reasoning**: Superior performance on complex queries
- **HuggingFace Models**: Support for CLaRa-7B-Base, CLaRa-7B-Instruct, CLaRa-7B-E2E

#### Forward Compatibility
- **Stable Contracts**: `Query`, `Answer`, `Provenance` won't change
- **Paradigm Agnostic**: Add new engines without modifying core code
- **Your Code Won't Break**: Upgrade to new engines with minimal changes

---

### 🚀 Added

#### Core Contracts (`fitz/core/`)
- `KnowledgeEngine` protocol - paradigm-agnostic engine interface
- `Query` dataclass - standardized query representation with constraints
- `Answer` dataclass - standardized response with provenance
- `Provenance` dataclass - source attribution for answers
- `Constraints` dataclass - query-time limits (max_sources, filters)
- Exception hierarchy: `QueryError`, `KnowledgeError`, `GenerationError`, `ConfigurationError`

#### Universal Runtime (`fitz/runtime/`)
- `run(query, engine="...")` - universal entry point
- `EngineRegistry` - global engine discovery and registration
- `create_engine(engine="...")` - factory for engine instances
- `list_engines()` - discover available engines
- `list_engines_with_info()` - engines with descriptions
- `@EngineRegistry.register_engine` - decorator for registration

#### CLaRa Engine (`fitz/engines/clara/`)
- `ClaraEngine` - full implementation of CLaRa paradigm
- `run_clara()` - convenience function for quick queries
- `create_clara_engine()` - factory for reusable instances
- `ClaraConfig` - comprehensive configuration
- `ClaraModelConfig` - model loading options (quantization, device, dtype)
- `ClaraCompressionConfig` - compression rate settings (16x-128x)
- `ClaraRetrievalConfig` - latent space retrieval settings
- `ClaraGenerationConfig` - generation parameters
- Auto-registration with global engine registry

#### Classic RAG Engine (`fitz/engines/classic_rag/`)
- `ClassicRagEngine` - wrapper implementing `KnowledgeEngine`
- `run_classic_rag()` - convenience function
- `create_classic_rag_engine()` - factory function
- Auto-registration with global engine registry

---

### 🔄 Changed

#### Public API (BREAKING)

**Entry Points**:
```python
# OLD
from fitz.pipeline.pipeline.engine import RAGPipeline
result = RAGPipeline.from_config(config).run("query")

# NEW
from fitz import run
answer = run("query", engine="classic_rag")
# or
from fitz.engines.classic_rag import run_classic_rag
answer = run_classic_rag("query")
```

**Answer Format**:
- `RGSAnswer.answer` → `Answer.text`
- `RGSAnswer.sources` → `Answer.provenance`
- `source.chunk_id` → `provenance.source_id`
- `source.text` → `provenance.excerpt`
- `source.metadata` → `provenance.metadata`

**Import Paths**:
| Old Path | New Path |
|----------|----------|
| `fitz.pipeline.*` | `fitz.engines.classic_rag.*` |
| `fitz.core.llm.*` | `fitz.llm.*` |
| `fitz.core.embedding.*` | `fitz.llm.embedding.*` |
| `fitz.core.vector_db.*` | `fitz.vector_db.*` |

#### CLI Commands
- `fitz-pipeline query` - now uses universal runtime internally
- Added `--engine` flag to select engine
- Added `fitz engines` command to list available engines

---

### 🏗️ Refactored

#### Directory Structure
```
OLD (v0.2.x):
fitz/
├── pipeline/          # RAG-specific
├── retrieval/         # RAG-specific
├── generation/        # RAG-specific
└── core/              # Mixed concerns

NEW (v0.3.0):
fitz/
├── core/              # Paradigm-agnostic contracts
├── engines/
│   ├── classic_rag/   # Traditional RAG
│   └── clara/         # CLaRa engine
├── runtime/           # Multi-engine orchestration
├── llm/               # Shared LLM service
├── vector_db/         # Shared vector DB service
└── ingest/            # Shared ingestion
```

#### Module Organization
- **Moved**: All RAG-specific code → `engines/classic_rag/`
- **Added**: CLaRa engine in `engines/clara/`
- **Promoted**: Shared services to root level
- **Created**: `runtime/` for multi-engine orchestration
- **Cleaned**: `core/` now contains only paradigm-agnostic contracts

---

### 🐛 Fixed
- Resolved all circular import dependencies
- Fixed import path inconsistencies across modules
- Corrected Provenance field usage (score → metadata)
- Fixed engine registration order to prevent import errors
- Proper lazy imports in runtime to avoid circular dependencies

---

### 📚 Documentation
- **New**: Complete README rewrite with multi-engine focus
- **New**: MIGRATION.md with detailed upgrade instructions
- **New**: ENGINES.md explaining engine architecture
- **New**: CUSTOM_ENGINES.md tutorial for creating engines
- **Updated**: All code examples to use new API
- **Added**: CLaRa-specific documentation and examples

---

### 🧪 Testing
- ✅ All existing tests updated and passing
- ✅ New tests for CLaRa engine (config, engine, runtime, registration)
- ✅ New tests for universal runtime
- ✅ New tests for engine registry
- ✅ Integration tests for multi-engine scenarios
- ✅ 155 tests total, all passing

---

### ⚠️ Breaking Changes Summary

1. **Import paths changed** - Update all imports (see Migration Guide)
2. **Public API changed** - Use `run()` or engine-specific functions
3. **Answer format changed** - `Answer.text` and `Answer.provenance`
4. **No backwards compatibility layer** - Clean break for cleaner codebase

---

### 📦 Dependencies

#### New Optional Dependencies
```toml
[project.optional-dependencies]
clara = ["transformers>=4.35.0", "torch>=2.0.0"]
```

#### Installation
```bash
# Base installation
pip install fitz

# With CLaRa support
pip install fitz[clara]

# Full installation
pip install fitz[all]
```

---

### 🔮 Future Compatibility

This architecture enables:
- **GraphRAG engine** - Knowledge graph-based retrieval (planned v0.3.1)
- **Engine composition** - Combine engines for ensemble (planned v0.4.0)
- **Streaming responses** - Real-time generation (planned v0.5.0)
- **Custom engines** - Users can implement and register their own

---

### 📖 Migration Path

See [MIGRATION.md](docs/MIGRATION.md) for detailed upgrade instructions.

**Quick migration**:
1. Update imports: `fitz.pipeline` → `fitz.engines.classic_rag`
2. Replace `RAGPipeline.run()` with `run_classic_rag()`
3. Update answer access: `result.answer` → `answer.text`
4. Update sources: `result.sources` → `answer.provenance`
5. Run tests to verify

---

## [0.2.0] - 2024-12-16

### Overview
Quality-focused release with enhanced observability, local-first development, and production readiness improvements.

### Added
- Contract Map tool for architecture documentation
- Ollama integration for local LLMs
- FAISS support for local vector database
- Enhanced error messages in API clients

### Improved
- Error handling with comprehensive logging
- Type safety (92% clean)
- API error messages with better context

---

## [0.1.0] - 2024-12-01

### Overview
Initial release of Fitz RAG framework.

### Added
- Core RAG pipeline
- OpenAI, Azure, Cohere LLM plugins
- Qdrant vector database integration
- Document ingestion pipeline
- CLI tools for query and ingestion
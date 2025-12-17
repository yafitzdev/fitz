# CHANGELOG.md
# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-01-17

### 🐛 Fixed

- **CLI Import Error**: Fixed misleading error messages when internal fitz modules fail to import
- **Detection Module**: Moved `fitz/cli/detect.py` to `fitz/core/detect.py` as single source of truth for service detection
- **FAISS Detection**: `SystemStatus.faiss` now returns `ServiceStatus` instead of boolean for consistent API
- **Registry Exceptions**: `LLMRegistryError` now inherits from `PluginNotFoundError` for consistent exception handling
- **Invalid Plugin Type**: `get_llm_plugin()` now raises `ValueError` for invalid plugin types (not just unknown plugins)
- **Ingest CLI**: Fixed import of non-existent `available_embedding_plugins` - now uses `available_llm_plugins("embedding")`
- **UTF-8 Encoding**: Added encoding declaration to handle emoji characters in error messages on Windows

### 🔧 Changed

- `fitz/core/detect.py` is now the canonical location for all service detection (Ollama, Qdrant, FAISS, API keys)
- `SystemStatus` now has `best_llm`, `best_embedding`, `best_vector_db` helper properties
- CLI modules (`doctor.py`, `init.py`) now import from `fitz.core.detect` instead of `fitz.cli.detect`

---

## [0.3.0] - 2025-12-17

### 🎉 Overview

Fitz v0.3.0 transforms the project from a RAG framework into a **multi-engine knowledge platform**. This release introduces a pluggable engine architecture, the CLaRa engine for compression-native RAG, and a universal runtime for seamless engine switching.

---

### ✨ Highlights

#### Multi-Engine Architecture
- **Universal Runtime**: `run(query, engine="clara")` - switch engines with one parameter
- **Engine Registry**: Discover, register, and manage knowledge engines
- **Protocol-Based Design**: Implement `answer(Query) -> Answer` to create custom engines
- **Shared Infrastructure**: LLM, vector DB, and ingestion services shared across engines

#### CLaRa Engine (NEW - Experimental)
- **Apple's CLaRa Integration**: Continuous Latent Reasoning for RAG
- **16x-128x Document Compression**: Preserve semantics while drastically reducing context
- **Unified Retrieval-Generation**: Single model for both retrieval and generation
- **Multi-Hop Reasoning**: Superior performance on complex queries
- **HuggingFace Models**: Support for CLaRa-7B-Base, CLaRa-7B-Instruct, CLaRa-7B-E2E

> ⚠️ **Note**: CLaRa requires significant hardware (16GB+ VRAM GPU recommended). 
> The engine is fully implemented and tested, but live inference requires the 7B model.

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
- 17 passing tests covering all functionality

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
from fitz.engines.classic_rag import run_classic_rag
answer = run_classic_rag("query")
```

**Answer Format**:
- `RGSAnswer.answer` → `Answer.text`
- `RGSAnswer.sources` → `Answer.provenance`
- `source.chunk_id` → `provenance.source_id`
- `source.text` → `provenance.excerpt`

#### CLI Commands
- `fitz-pipeline query` - now uses universal runtime internally
- Added `--engine` flag to select engine (future)
- Enhanced error messages with exception hierarchy

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

---

### 🐛 Fixed
- Resolved all circular import dependencies
- Fixed import path inconsistencies across modules
- Corrected Provenance field usage (score → metadata)
- Fixed engine registration order to prevent import errors
- Proper lazy imports in runtime to avoid circular dependencies

---

### 📚 Documentation
- Updated README with multi-engine architecture
- Added CLaRa hardware requirements
- Migration guide for v0.2.x → v0.3.0
- Updated all code examples

---

### 🧪 Testing
- ✅ All existing tests updated and passing
- ✅ 17 new tests for CLaRa engine (config, engine, runtime, registration)
- ✅ Tests use mocked dependencies (no GPU required for testing)
- ✅ Integration tests for engine protocol compliance

---

### ⚠️ Breaking Changes Summary

1. **Import paths changed** - Update all imports (see Migration Guide)
2. **Public API changed** - Use `run_classic_rag()` or engine-specific functions
3. **Answer format changed** - `Answer.text` and `Answer.provenance`
4. **No backwards compatibility layer** - Clean break for cleaner codebase

---

### 📦 Dependencies

#### New Optional Dependencies
```toml
[project.optional-dependencies]
clara = ["transformers>=4.35.0", "torch>=2.0.0"]
```

---

### 🔮 Future Compatibility

This architecture enables:
- **GraphRAG engine** - Knowledge graph-based retrieval (planned)
- **CLaRa MLX** - Smaller models for Apple Silicon (when available)
- **Engine composition** - Combine engines for ensemble approaches
- **Custom engines** - Users can implement and register their own

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
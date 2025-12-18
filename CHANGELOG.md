# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

---

## [0.3.3] - 2025-12-19

### 🎉 Highlights

**YAML-based Plugin System** - LLM and Vector DB plugins are now defined entirely in YAML, not Python. Adding new providers is now as simple as creating a YAML file.

### 🚀 Added

- **YAML-based LLM plugins**: Chat, Embedding, and Rerank plugins now use YAML specs
  - `fitz/llm/chat/*.yaml` - Chat plugins (OpenAI, Cohere, Anthropic, Azure, Ollama)
  - `fitz/llm/embedding/*.yaml` - Embedding plugins  
  - `fitz/llm/rerank/*.yaml` - Rerank plugins
- **YAML-based Vector DB plugins**: Vector databases now use YAML specs
  - `fitz/vector_db/plugins/qdrant.yaml`
  - `fitz/vector_db/plugins/pinecone.yaml`
  - `fitz/vector_db/plugins/local_faiss.yaml`
- **Generic plugin runtime**: `GenericVectorDBPlugin` and `YAMLPluginBase` execute YAML specs at runtime
- **Provider-agnostic features**: YAML `features` section for provider-specific behavior
  - `requires_uuid_ids`: Auto-convert string IDs to UUIDs
  - `auto_detect`: Service discovery configuration
- **Message transforms**: Pluggable message format transformers for different LLM APIs
  - `openai_chat`, `cohere_chat`, `anthropic_chat`, `ollama_chat`, `gemini_chat`

### 🔄 Changed

- **LLM plugins**: Migrated from Python classes to YAML specifications
- **Vector DB plugins**: Migrated from Python classes to YAML specifications  
- **Plugin discovery**: Now scans `*.yaml` files instead of `*.py` modules
- **fitz/core/registry.py**: Single source of truth for all plugin access

### 🐛 Fixed

- **Qdrant 400 Bad Request**: String IDs now converted to UUIDs automatically
- **Auto-create collection**: Collections created on first upsert (handles 404)
- **Import errors in CLI**: Fixed by adding re-exports to `fitz/core/registry.py`

---

## [0.3.2] - 2025-12-18

### 🔄 Changed

- Renamed config field `llm` → `chat` for clarity (breaking change - regenerate config with `fitz init`)

### 🚀 Added

- `fitz db` command to inspect vector database collections
- `fitz chunk` command to preview chunking strategies
- `fitz query` as top-level command (was `fitz pipeline query`)
- `fitz config` as top-level command (was `fitz pipeline config show`)
- LAN scanning for Qdrant detection in `fitz init`
- Auto-select single provider options in `fitz init`

### 🐛 Fixed

- Contract map now discovers re-exported plugins (local-faiss)
- Contract map health check false positives removed
- Test fixes for `llm` → `chat` rename

---

## [0.3.1] - 2025-01-17

### 🐛 Fixed

- **CLI Import Error**: Fixed misleading error messages when internal fitz modules fail to import
- **Detection Module**: Moved `fitz/cli/detect.py` to `fitz/core/detect.py` as single source of truth for service detection
- **FAISS Detection**: `SystemStatus.faiss` now returns `ServiceStatus` instead of boolean for consistent API
- **Registry Exceptions**: `LLMRegistryError` now inherits from `PluginNotFoundError` for consistent exception handling
- **Invalid Plugin Type**: `get_llm_plugin()` now raises `ValueError` for invalid plugin types (not just unknown plugins)
- **Ingest CLI**: Fixed import of non-existent `available_embedding_plugins` now uses `available_llm_plugins("embedding")`
- **UTF-8 Encoding**: Added encoding declaration to handle emoji characters in error messages on Windows

### 🔄 Changed

- `fitz/core/detect.py` is now the canonical location for all service detection (Ollama, Qdrant, FAISS, API keys)
- `SystemStatus` now has `best_llm`, `best_embedding`, `best_vector_db` helper properties
- CLI modules (`doctor.py`, `init.py`) now import from `fitz.core.detect` instead of `fitz.cli.detect`

---

## [0.3.0] - 2024-12-17

### 🎉 Overview

Fitz v0.3.0 transforms the project from a RAG framework into a **multi-engine knowledge platform**. This release introduces a pluggable engine architecture, the CLaRa engine for compression-native RAG, and a universal runtime for seamless engine switching.

### ✨ Highlights

- **Universal Runtime**: `run(query, engine="clara")` switch engines with one parameter
- **Engine Registry**: Discover, register, and manage knowledge engines
- **Protocol-Based Design**: Implement `answer(Query) -> Answer` to create custom engines
- **CLaRa Engine**: Apple's Continuous Latent Reasoning with 16x-128x document compression

### 🚀 Added

#### Core Contracts (`fitz/core/`)
- `KnowledgeEngine` protocol for paradigm-agnostic engine interface
- `Query` dataclass for standardized query representation with constraints
- `Answer` dataclass for standardized response with provenance
- `Provenance` dataclass for source attribution
- `Constraints` dataclass for query-time limits (max_sources, filters)
- Exception hierarchy: `QueryError`, `KnowledgeError`, `GenerationError`, `ConfigurationError`

#### Universal Runtime (`fitz/runtime/`)
- `run(query, engine="...")` universal entry point
- `EngineRegistry` for global engine discovery and registration
- `create_engine(engine="...")` factory for engine instances
- `list_engines()` to discover available engines
- `list_engines_with_info()` for engines with descriptions

#### CLaRa Engine (`fitz/engines/clara/`)
- `ClaraEngine` full implementation of CLaRa paradigm
- `run_clara()` convenience function for quick queries
- `create_clara_engine()` factory for reusable instances
- `ClaraConfig` comprehensive configuration
- Auto-registration with global engine registry
- 17 passing tests covering all functionality

#### Classic RAG Engine (`fitz/engines/classic_rag/`)
- `ClassicRagEngine` wrapper implementing `KnowledgeEngine`
- `run_classic_rag()` convenience function
- `create_classic_rag_engine()` factory function
- Auto-registration with global engine registry

### 🔄 Changed

#### Public API (BREAKING)
- Entry points: `RAGPipeline.from_config(config).run()` → `run_classic_rag()`
- Answer format: `RGSAnswer.answer` → `Answer.text`
- Source format: `RGSAnswer.sources` → `Answer.provenance`
- Chunk ID: `source.chunk_id` → `provenance.source_id`
- Text excerpt: `source.text` → `provenance.excerpt`

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

### 🐛 Fixed

- Resolved all circular import dependencies
- Fixed import path inconsistencies across modules
- Corrected Provenance field usage (score → metadata)
- Fixed engine registration order to prevent import errors
- Proper lazy imports in runtime to avoid circular dependencies

### 📚 Documentation

- Updated README with multi-engine architecture
- Added CLaRa hardware requirements
- Migration guide for v0.2.x → v0.3.0
- Updated all code examples

### 🧪 Testing

- All existing tests updated and passing
- 17 new tests for CLaRa engine (config, engine, runtime, registration)
- Tests use mocked dependencies (no GPU required for testing)
- Integration tests for engine protocol compliance

### ⚠️ Breaking Changes

1. **Import paths changed**: Update all imports (see Migration Guide)
2. **Public API changed**: Use `run_classic_rag()` or engine-specific functions
3. **Answer format changed**: `Answer.text` and `Answer.provenance`
4. **No backwards compatibility layer**: Clean break for cleaner codebase

### 📦 Dependencies

New optional dependencies:
```toml
[project.optional-dependencies]
clara = ["transformers>=4.35.0", "torch>=2.0.0"]
```

---

## [0.2.0] - 2025-12-16

### 🎉 Overview

Quality-focused release with enhanced observability, local-first development, and production readiness improvements.

### ✨ Highlights

- **Contract Map Tool**: Living architecture documentation with automatic quality tracking
- **Ollama Integration**: Use local LLMs (Llama, Mistral, etc.) with zero API costs
- **FAISS Support**: Local vector database for development and testing
- **Production Readiness**: 100% appropriate error handling, zero architecture violations

### 🚀 Added

#### Quality Tools
- Contract map with Any usage analysis
- Exception pattern detection
- Code quality metrics tracking
- Architecture violation detection

#### Local Runtime
- Ollama backend for chat, embedding, rerank
- FAISS local vector database
- Local development workflow

#### Developer Experience
- Enhanced error messages in API clients
- Improved logging for file operations
- Better type hints throughout
- Comprehensive documentation

### 🔄 Changed

- Error handling with comprehensive logging
- Type safety improved (92% clean)
- API error messages with better context

### 📚 Documentation

- Updated README with v0.2.0 features
- Contract Map tool documentation
- Local development guide

---

## [0.1.0] - 2025-12-14

### 🎉 Overview

Initial release of Fitz RAG framework.

### 🚀 Added

- Core RAG pipeline
- OpenAI, Azure, Cohere LLM plugins
- Qdrant vector database integration
- Document ingestion pipeline
- CLI tools for query and ingestion

---

[Unreleased]: https://github.com/yafitzdev/fitz/compare/v0.3.3...HEAD
[0.3.3]: https://github.com/yafitzdev/fitz/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/yafitzdev/fitz/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/yafitzdev/fitz/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/yafitzdev/fitz/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/yafitzdev/fitz/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/yafitzdev/fitz/releases/tag/v0.1.0
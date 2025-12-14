# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

---

## [0.1.0] - 2025-12-14

### Added

#### Core Framework (`core/`)
- **Plugin Registry System**: Centralized, auto-discovering plugin architecture
  - Support for `chat`, `embedding`, `rerank`, and `vector_db` plugin types
  - Protocol-based contracts with runtime validation
  - Lazy discovery via `pkgutil.iter_modules()`
- **LLM Abstractions**
  - `ChatPlugin` protocol and `ChatEngine` wrapper
  - `EmbeddingPlugin` protocol and `EmbeddingEngine` wrapper
  - `RerankPlugin` protocol and `RerankEngine` wrapper
  - Cohere implementations for all three plugin types
- **Vector Database Layer**
  - `VectorDBPlugin` protocol with Qdrant implementation
  - `VectorDBWriter` for chunk upsert with deduplication hashing
  - `VectorDBEngine` for search operations
- **Canonical Data Models**
  - `Chunk`: Universal chunk representation across the stack
  - `Document`: Document-level metadata container
- **Configuration System**
  - Pydantic-based config schemas with strict validation
  - YAML config loading with environment variable expansion
  - Centralized credential resolution for LLM providers
- **Unified Logging**
  - Consistent log format across all modules
  - Subsystem tags for searchable log output

#### RAG Pipeline (`rag/`)
- **Pipeline Engine**
  - `RAGPipeline`: Main orchestration class
  - Config-driven pipeline construction via `from_config()`
  - Pipeline plugins: `standard`, `fast`, `debug`, `easy`
- **Retrieval System**
  - `RetrievalPlugin` protocol
  - `DenseRetrievalPlugin`: Vector similarity search with optional reranking
  - `RetrieverEngine`: Orchestration layer with registry integration
- **Context Processing Pipeline**
  - Step-based architecture: normalize → dedupe → group → merge → pack
  - `NormalizeStep`: Canonicalize heterogeneous chunk inputs
  - `DedupeStep`: Remove duplicate content
  - `GroupByDocumentStep`: Organize chunks by source document
  - `MergeAdjacentStep`: Combine consecutive chunks
  - `PackWindowStep`: Fit chunks within token/character budget
- **RGS (Retrieval-Guided Synthesis)**
  - Configurable prompt assembly with slot system
  - Citation support with source labels
  - Strict grounding mode for hallucination prevention
  - Answer synthesis with source tracking
- **CLI**
  - `fitz-rag config show`: Display effective configuration
  - `fitz-rag query`: Run one-off RAG queries

#### Ingestion Pipeline (`ingest/`)
- **Ingestion Engine**
  - `IngestPlugin` protocol
  - `LocalFSIngestPlugin`: Local filesystem document ingestion
  - `IngestionEngine`: Config-driven ingestion orchestration
- **Chunking System**
  - `ChunkerPlugin` protocol
  - `SimpleChunker`: Fixed-size character chunking
  - `ChunkingEngine`: Chunking orchestration with validation
- **Document Validation**
  - Filter empty/whitespace documents
  - Configurable minimum content length
- **Ingestion Pipeline**
  - End-to-end: ingest → chunk → embed → write to vector DB
- **CLI**
  - `fitz-ingest run`: Ingest documents into vector database

#### Development Tools (`tools/`)
- **Contract Map Generator**
  - Extract and document models, protocols, registries
  - Import graph analysis with layering violation detection
  - Plugin discovery report across all namespaces
  - Code statistics and hotspot analysis
  - Config surface mapping

### Technical Details
- Python 3.12+ required
- Dependencies: pydantic, pyyaml, qdrant-client, httpx, typing-extensions
- Optional: cohere, pdfminer.six, python-docx

---

[Unreleased]: https://github.com/yafitzdev/fitz/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/yafitzdev/fitz/releases/tag/v0.1.0

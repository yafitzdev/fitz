# tools/smoketest/smoke_classic_rag_e2e.py
"""
End-to-End Smoketest for Classic RAG Engine.

Tests the complete Classic RAG pipeline with mocked external services:
1. Document ingestion (reading files)
2. Chunking (splitting documents)
3. Embedding (creating vectors)
4. Vector storage (upserting to DB)
5. Retrieval (searching)
6. Reranking (optional)
7. RGS prompt building
8. LLM generation
9. Answer formatting with provenance

This test uses in-memory mocks so it can run without:
- Ollama
- OpenAI/Cohere API keys
- Qdrant/external vector DB

Usage:
    python tools/smoketest/smoke_classic_rag_e2e.py
    python tools/smoketest/smoke_classic_rag_e2e.py --with-files ./test_docs
    python tools/smoketest/smoke_classic_rag_e2e.py --verbose
"""

from __future__ import annotations

import argparse
import math
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

# =============================================================================
# Test Configuration
# =============================================================================


@dataclass
class TestConfig:
    """Configuration for the smoketest."""

    docs_path: Optional[Path] = None
    verbose: bool = False
    embedding_dim: int = 64
    top_k: int = 3
    max_chunks: int = 5


# =============================================================================
# Mock Components (No External Dependencies)
# =============================================================================


def deterministic_embed(text: str, dim: int = 64) -> List[float]:
    """
    Create a deterministic embedding from text.

    This creates vectors that have meaningful similarity:
    similar texts will have similar vectors.
    """
    vec = [0.0] * dim
    text_lower = text.lower()

    # Use character n-grams to create semantic-ish embeddings
    for i in range(len(text_lower) - 2):
        trigram = text_lower[i : i + 3]
        hash_val = hash(trigram) % dim
        vec[hash_val] += 1.0

    # Add word-level features
    for word in text_lower.split():
        hash_val = hash(word) % dim
        vec[hash_val] += 2.0

    # Normalize
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)) or 1.0
    norm_b = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (norm_a * norm_b)


@dataclass
class MockSearchResult:
    """Mock search result from vector DB."""

    id: str
    payload: dict
    score: float


class MockVectorDB:
    """In-memory vector database for testing."""

    def __init__(self):
        self._collections: dict[str, list[dict]] = {}

    def create_collection(self, name: str) -> None:
        if name not in self._collections:
            self._collections[name] = []

    def upsert(self, collection: str, points: list[dict]) -> None:
        self.create_collection(collection)
        # Replace by ID
        existing = {p["id"]: p for p in self._collections[collection]}
        for p in points:
            existing[p["id"]] = p
        self._collections[collection] = list(existing.values())

    def search(
        self, collection_name: str, query_vector: list[float], limit: int, with_payload: bool = True
    ) -> list[MockSearchResult]:
        if collection_name not in self._collections:
            return []

        results = []
        for point in self._collections[collection_name]:
            score = cosine_similarity(query_vector, point["vector"])
            results.append(
                MockSearchResult(
                    id=str(point["id"]), payload=dict(point.get("payload", {})), score=score
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    def count(self, collection: str) -> int:
        return len(self._collections.get(collection, []))


class MockEmbedder:
    """Mock embedding engine."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.call_count = 0

    def embed(self, text: str) -> list[float]:
        self.call_count += 1
        return deterministic_embed(text, self.dim)


class MockLLM:
    """Mock LLM for testing."""

    def __init__(self):
        self.call_count = 0
        self.last_messages: list[dict] = []

    def chat(self, messages: list[dict]) -> str:
        self.call_count += 1
        self.last_messages = messages

        # Extract question from user message
        user_msg = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        # Generate a mock answer that references sources
        return (
            "Based on the provided sources, here is the answer: "
            "The documents discuss various topics including technology and science. "
            "[S1] provides key context, while [S2] offers supporting details."
        )


class MockReranker:
    """Mock reranker for testing."""

    def __init__(self):
        self.call_count = 0

    def rerank(self, query: str, chunks: list) -> list:
        self.call_count += 1
        # Simple rerank: boost chunks that contain query words
        query_words = set(query.lower().split())

        def score(chunk) -> float:
            content_words = set(chunk.content.lower().split())
            overlap = len(query_words & content_words)
            return overlap

        return sorted(chunks, key=score, reverse=True)


# =============================================================================
# Test Documents
# =============================================================================

SAMPLE_DOCUMENTS = [
    {
        "id": "doc1",
        "title": "Introduction to Machine Learning",
        "content": """
Machine learning is a subset of artificial intelligence that enables computers 
to learn from data without being explicitly programmed. It uses algorithms to 
find patterns in data and make predictions or decisions.

There are three main types of machine learning:
1. Supervised learning - learns from labeled data
2. Unsupervised learning - finds patterns in unlabeled data  
3. Reinforcement learning - learns through trial and error

Popular machine learning frameworks include TensorFlow, PyTorch, and scikit-learn.
These tools make it easier to build and train models.
""",
    },
    {
        "id": "doc2",
        "title": "Neural Networks Explained",
        "content": """
Neural networks are computing systems inspired by biological neural networks 
in the human brain. They consist of layers of interconnected nodes (neurons) 
that process information.

A typical neural network has:
- Input layer: receives the initial data
- Hidden layers: process and transform the data
- Output layer: produces the final result

Deep learning refers to neural networks with many hidden layers. These deep 
networks can learn complex patterns and are used for image recognition, 
natural language processing, and many other tasks.
""",
    },
    {
        "id": "doc3",
        "title": "Vector Databases Overview",
        "content": """
Vector databases are specialized databases designed to store and query 
high-dimensional vector embeddings. They are essential for modern AI 
applications like semantic search and recommendation systems.

Key features of vector databases:
- Efficient similarity search using algorithms like HNSW and IVF
- Support for filtering and hybrid search
- Scalable to billions of vectors

Popular vector databases include Qdrant, Pinecone, Milvus, and Weaviate.
RAG (Retrieval-Augmented Generation) systems heavily rely on vector databases 
to find relevant context for language models.
""",
    },
    {
        "id": "doc4",
        "title": "RAG Architecture",
        "content": """
RAG (Retrieval-Augmented Generation) is an architecture that combines 
information retrieval with language model generation. It addresses the 
limitation of LLMs having static knowledge.

The RAG pipeline typically involves:
1. Query embedding: convert the user question to a vector
2. Retrieval: find relevant documents using vector similarity
3. Context building: format retrieved documents as context
4. Generation: use an LLM to generate an answer based on context

RAG improves accuracy by grounding responses in retrieved facts, 
reducing hallucination and providing citations.
""",
    },
]


def create_test_files(base_dir: Path) -> List[Path]:
    """Create test document files."""
    files = []
    for doc in SAMPLE_DOCUMENTS:
        file_path = base_dir / f"{doc['id']}.txt"
        content = f"# {doc['title']}\n\n{doc['content']}"
        file_path.write_text(content, encoding="utf-8")
        files.append(file_path)
    return files


# =============================================================================
# Pipeline Components Under Test
# =============================================================================


def document_ingestion(config: TestConfig) -> tuple[list[dict], bool]:
    """Test 1: Document ingestion."""
    print("\n" + "=" * 60)
    print("TEST 1: DOCUMENT INGESTION")
    print("=" * 60)

    documents = []

    if config.docs_path and config.docs_path.exists():
        # Read from actual files
        for path in config.docs_path.rglob("*"):
            if path.suffix.lower() in {".txt", ".md"}:
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    documents.append({"id": str(path.stem), "path": str(path), "content": content})
                except Exception as e:
                    print(f"  ⚠ Failed to read {path}: {e}")
    else:
        # Use sample documents
        for doc in SAMPLE_DOCUMENTS:
            documents.append(
                {
                    "id": doc["id"],
                    "path": f"sample/{doc['id']}.txt",
                    "content": doc["content"],
                    "title": doc["title"],
                }
            )

    print(f"  ✓ Loaded {len(documents)} documents")
    if config.verbose:
        for doc in documents:
            preview = doc["content"][:50].replace("\n", " ")
            print(f"    - {doc['id']}: {preview}...")

    return documents, len(documents) > 0


def chunking(documents: list[dict], config: TestConfig) -> tuple[list[dict], bool]:
    """Test 2: Document chunking."""
    print("\n" + "=" * 60)
    print("TEST 2: DOCUMENT CHUNKING")
    print("=" * 60)

    from fitz.engines.classic_rag.models.chunk import Chunk

    chunks = []
    for doc in documents:
        # Split by paragraphs (simple chunking)
        paragraphs = [p.strip() for p in doc["content"].split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            if len(para) < 20:  # Skip very short chunks
                continue
            chunks.append(
                {
                    "id": f"{doc['id']}:{i}",
                    "doc_id": doc["id"],
                    "chunk_index": i,
                    "content": para,
                    "metadata": {"path": doc.get("path", ""), "title": doc.get("title", doc["id"])},
                }
            )

    print(f"  ✓ Created {len(chunks)} chunks from {len(documents)} documents")
    if config.verbose:
        for chunk in chunks[:3]:
            preview = chunk["content"][:40].replace("\n", " ")
            print(f"    - {chunk['id']}: {preview}...")

    return chunks, len(chunks) > 0


def embedding(
    chunks: list[dict], embedder: MockEmbedder, config: TestConfig
) -> tuple[list[list[float]], bool]:
    """Test 3: Embedding generation."""
    print("\n" + "=" * 60)
    print("TEST 3: EMBEDDING GENERATION")
    print("=" * 60)

    vectors = []
    for chunk in chunks:
        vec = embedder.embed(chunk["content"])
        vectors.append(vec)

    print(f"  ✓ Generated {len(vectors)} embeddings")
    print(f"  ✓ Embedding dimension: {len(vectors[0]) if vectors else 0}")
    print(f"  ✓ Embedder called {embedder.call_count} times")

    if config.verbose and len(vectors) >= 2:
        # Show similarity between first two chunks
        sim = cosine_similarity(vectors[0], vectors[1])
        print(f"    - Similarity(chunk0, chunk1): {sim:.4f}")

    return vectors, len(vectors) == len(chunks)


def vector_storage(
    chunks: list[dict], vectors: list[list[float]], vector_db: MockVectorDB, config: TestConfig
) -> bool:
    """Test 4: Vector storage (upsert)."""
    print("\n" + "=" * 60)
    print("TEST 4: VECTOR STORAGE")
    print("=" * 60)

    collection = "test_collection"

    # Prepare points for upsert
    points = []
    for chunk, vector in zip(chunks, vectors):
        points.append(
            {
                "id": chunk["id"],
                "vector": vector,
                "payload": {
                    "doc_id": chunk["doc_id"],
                    "chunk_index": chunk["chunk_index"],
                    "content": chunk["content"],
                    **chunk.get("metadata", {}),
                },
            }
        )

    vector_db.upsert(collection, points)

    count = vector_db.count(collection)
    print(f"  ✓ Upserted {len(points)} vectors to '{collection}'")
    print(f"  ✓ Collection now has {count} vectors")

    return count == len(chunks)


def retrieval(
    vector_db: MockVectorDB, embedder: MockEmbedder, config: TestConfig
) -> tuple[list[dict], bool]:
    """Test 5: Retrieval (search)."""
    print("\n" + "=" * 60)
    print("TEST 5: RETRIEVAL")
    print("=" * 60)

    query = "What is RAG and how does it work with vector databases?"
    collection = "test_collection"

    # Embed query
    query_vector = embedder.embed(query)

    # Search
    results = vector_db.search(collection, query_vector, limit=config.top_k)

    print(f"  ✓ Query: '{query}'")
    print(f"  ✓ Retrieved {len(results)} results")

    retrieved_chunks = []
    for i, result in enumerate(results):
        retrieved_chunks.append({"id": result.id, "score": result.score, **result.payload})
        if config.verbose:
            preview = result.payload.get("content", "")[:40].replace("\n", " ")
            print(f"    {i + 1}. [{result.score:.4f}] {result.id}: {preview}...")

    return retrieved_chunks, len(results) > 0


def reranking(
    chunks: list[dict], reranker: MockReranker, config: TestConfig
) -> tuple[list[dict], bool]:
    """Test 6: Reranking (optional)."""
    print("\n" + "=" * 60)
    print("TEST 6: RERANKING")
    print("=" * 60)

    from fitz.engines.classic_rag.models.chunk import Chunk

    query = "What is RAG and how does it work with vector databases?"

    # Convert to Chunk objects
    chunk_objects = [
        Chunk(
            id=c["id"],
            doc_id=c["doc_id"],
            chunk_index=c.get("chunk_index", 0),
            content=c["content"],
            metadata=c.get("metadata", {}),
        )
        for c in chunks
    ]

    # Rerank
    reranked = reranker.rerank(query, chunk_objects)

    print(f"  ✓ Reranked {len(reranked)} chunks")
    print(f"  ✓ Reranker called {reranker.call_count} times")

    # Convert back to dicts
    reranked_dicts = [
        {"id": c.id, "doc_id": c.doc_id, "content": c.content, "metadata": c.metadata}
        for c in reranked
    ]

    if config.verbose:
        for i, c in enumerate(reranked[:3]):
            preview = c.content[:40].replace("\n", " ")
            print(f"    {i + 1}. {c.id}: {preview}...")

    return reranked_dicts, len(reranked) > 0


def rgs_prompt_building(chunks: list[dict], config: TestConfig) -> tuple[dict, bool]:
    """Test 7: RGS prompt building."""
    print("\n" + "=" * 60)
    print("TEST 7: RGS PROMPT BUILDING")
    print("=" * 60)

    from fitz.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSConfig
    from fitz.engines.classic_rag.models.chunk import Chunk

    query = "What is RAG and how does it work?"

    # Convert to Chunk objects
    chunk_objects = [
        Chunk(
            id=c["id"],
            doc_id=c["doc_id"],
            chunk_index=c.get("chunk_index", 0),
            content=c["content"],
            metadata=c.get("metadata", {}),
        )
        for c in chunks[: config.max_chunks]
    ]

    # Build RGS prompt
    rgs = RGS(RGSConfig(enable_citations=True, strict_grounding=True, max_chunks=config.max_chunks))

    prompt = rgs.build_prompt(query, chunk_objects)

    print(f"  ✓ Built RGS prompt with {len(chunk_objects)} chunks")
    print(f"  ✓ System prompt length: {len(prompt.system)} chars")
    print(f"  ✓ User prompt length: {len(prompt.user)} chars")

    if config.verbose:
        print("\n  --- System Prompt Preview ---")
        print(f"  {prompt.system[:200]}...")
        print("\n  --- User Prompt Preview ---")
        print(f"  {prompt.user[:200]}...")

    return {"system": prompt.system, "user": prompt.user}, bool(prompt.system and prompt.user)


def llm_generation(prompt: dict, llm: MockLLM, config: TestConfig) -> tuple[str, bool]:
    """Test 8: LLM generation."""
    print("\n" + "=" * 60)
    print("TEST 8: LLM GENERATION")
    print("=" * 60)

    messages = [
        {"role": "system", "content": prompt["system"]},
        {"role": "user", "content": prompt["user"]},
    ]

    response = llm.chat(messages)

    print(f"  ✓ Generated response ({len(response)} chars)")
    print(f"  ✓ LLM called {llm.call_count} times")

    if config.verbose:
        print(f"\n  --- Response ---")
        print(f"  {response}")

    return response, bool(response)


def answer_formatting(response: str, chunks: list[dict], config: TestConfig) -> tuple[dict, bool]:
    """Test 9: Answer formatting with provenance."""
    print("\n" + "=" * 60)
    print("TEST 9: ANSWER FORMATTING")
    print("=" * 60)

    from fitz.core import Answer, Provenance

    # Build provenance from chunks
    provenance = []
    for i, chunk in enumerate(chunks[: config.max_chunks]):
        provenance.append(
            Provenance(
                source_id=chunk["id"],
                excerpt=chunk["content"][:100],
                metadata={
                    "doc_id": chunk["doc_id"],
                    "relevance_rank": i + 1,
                    **chunk.get("metadata", {}),
                },
            )
        )

    # Create Answer object
    answer = Answer(
        text=response,
        provenance=provenance,
        metadata={"engine": "classic_rag", "model": "mock_llm", "num_sources": len(provenance)},
    )

    print(f"  ✓ Created Answer with {len(answer.provenance)} sources")
    print(f"  ✓ Answer text: {answer.text[:50]}...")

    if config.verbose:
        print("\n  --- Provenance ---")
        for p in answer.provenance[:3]:
            print(f"    - {p.source_id}: {p.excerpt[:40]}...")

    return {
        "text": answer.text,
        "provenance": [{"source_id": p.source_id, "excerpt": p.excerpt} for p in answer.provenance],
        "metadata": answer.metadata,
    }, bool(answer.text and answer.provenance)


# =============================================================================
# Main Test Runner
# =============================================================================


def run_all_tests(config: TestConfig) -> bool:
    """Run all tests and return success status."""
    print("\n" + "=" * 60)
    print("CLASSIC RAG END-TO-END SMOKETEST")
    print("=" * 60)

    # Initialize mocks
    vector_db = MockVectorDB()
    embedder = MockEmbedder(dim=config.embedding_dim)
    reranker = MockReranker()
    llm = MockLLM()

    results = []

    # Test 1: Document ingestion
    documents, passed = document_ingestion(config)
    results.append(("Document Ingestion", passed))
    if not passed:
        print("  ✗ FAILED: No documents loaded")
        return False

    # Test 2: Chunking
    chunks, passed = chunking(documents, config)
    results.append(("Chunking", passed))
    if not passed:
        print("  ✗ FAILED: No chunks created")
        return False

    # Test 3: Embedding
    vectors, passed = embedding(chunks, embedder, config)
    results.append(("Embedding", passed))
    if not passed:
        print("  ✗ FAILED: Embedding failed")
        return False

    # Test 4: Vector storage
    passed = vector_storage(chunks, vectors, vector_db, config)
    results.append(("Vector Storage", passed))
    if not passed:
        print("  ✗ FAILED: Vector storage failed")
        return False

    # Test 5: Retrieval
    retrieved, passed = retrieval(vector_db, embedder, config)
    results.append(("Retrieval", passed))
    if not passed:
        print("  ✗ FAILED: Retrieval failed")
        return False

    # Test 6: Reranking
    reranked, passed = reranking(retrieved, reranker, config)
    results.append(("Reranking", passed))
    # Reranking is optional, so we continue even if it fails

    # Test 7: RGS prompt building
    prompt, passed = rgs_prompt_building(reranked, config)
    results.append(("RGS Prompt", passed))
    if not passed:
        print("  ✗ FAILED: Prompt building failed")
        return False

    # Test 8: LLM generation
    response, passed = llm_generation(prompt, llm, config)
    results.append(("LLM Generation", passed))
    if not passed:
        print("  ✗ FAILED: LLM generation failed")
        return False

    # Test 9: Answer formatting
    answer, passed = answer_formatting(response, reranked, config)
    results.append(("Answer Formatting", passed))
    if not passed:
        print("  ✗ FAILED: Answer formatting failed")
        return False

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("🎉 All tests passed!")
    else:
        print("❌ Some tests failed")

    return all_passed


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="End-to-end smoketest for Classic RAG engine")
    parser.add_argument(
        "--with-files",
        type=Path,
        help="Path to directory with test documents (uses sample docs if not provided)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument(
        "--top-k", type=int, default=3, help="Number of results to retrieve (default: 3)"
    )

    args = parser.parse_args()

    config = TestConfig(docs_path=args.with_files, verbose=args.verbose, top_k=args.top_k)

    success = run_all_tests(config)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

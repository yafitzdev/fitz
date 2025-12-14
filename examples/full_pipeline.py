#!/usr/bin/env python3
"""
fitz Full Pipeline Example

This example demonstrates the complete end-to-end RAG flow:
1. Ingest documents from local filesystem
2. Chunk and embed the content
3. Store in Qdrant vector database
4. Query with RAG pipeline

Prerequisites:
    1. Start Qdrant:
       docker run -p 6333:6333 qdrant/qdrant

    2. Set API key:
       export COHERE_API_KEY="your-key"

Run with:
    python examples/full_pipeline.py
"""

import os
import sys
from pathlib import Path

# Check prerequisites
if not os.getenv("COHERE_API_KEY"):
    print("ERROR: COHERE_API_KEY environment variable not set")
    print("Run: export COHERE_API_KEY='your-key'")
    sys.exit(1)

from fitz.core.llm import get_llm_plugin
from fitz.core.vector_db.writer import VectorDBWriter
from fitz.ingest.chunking.plugins.simple import SimpleChunker
from fitz.ingest.ingestion.registry import get_ingest_plugin
from fitz.ingest.validation.documents import ValidationConfig, validate
from fitz.pipeline.context.pipeline import ContextPipeline
from fitz.generation.retrieval_guided.synthesis import RGS, RGSConfig
from fitz.retrieval.runtime.plugins.dense import DenseRetrievalPlugin

COLLECTION_NAME = "fitz_demo"


def setup_test_documents() -> Path:
    """Create test documents if they don't exist."""
    test_dir = Path("test_docs")
    if not test_dir.exists():
        print("Creating test documents...")
        test_dir.mkdir(exist_ok=True)

        (test_dir / "rag_overview.txt").write_text(
            "RAG (Retrieval-Augmented Generation) is a technique that combines "
            "information retrieval with language model generation. The system first "
            "retrieves relevant documents from a knowledge base using semantic search, "
            "then uses those documents as context for generating accurate responses. "
            "This approach significantly reduces hallucinations and grounds the model's "
            "responses in factual, retrievable information."
        )

        (test_dir / "vector_databases.txt").write_text(
            "Vector databases are specialized storage systems designed to store and "
            "query high-dimensional embedding vectors. They enable semantic search by "
            "converting text into numerical vectors and finding similar vectors using "
            "distance metrics like cosine similarity. Popular vector databases include "
            "Qdrant, Pinecone, Weaviate, Milvus, and Chroma. These databases are essential "
            "components in modern RAG systems."
        )

        (test_dir / "chunking_strategies.txt").write_text(
            "Chunking is the process of splitting documents into smaller pieces for "
            "embedding and retrieval. Common strategies include fixed-size chunking, "
            "sentence-based chunking, and semantic chunking. The chunk size affects "
            "retrieval quality - smaller chunks are more precise but may lose context, "
            "while larger chunks preserve context but may include irrelevant information. "
            "Overlap between chunks helps maintain continuity across boundaries."
        )

        print(f"Created test documents in {test_dir}/")
    return test_dir


def ingest_documents(test_dir: Path) -> None:
    """Ingest, chunk, embed, and store documents."""
    print("\n" + "=" * 60)
    print("PHASE 1: Document Ingestion")
    print("=" * 60)

    # Step 1: Ingest raw documents
    print("\n[1/4] Ingesting documents...")
    LocalIngestPlugin = get_ingest_plugin("local")
    ingest_plugin = LocalIngestPlugin()
    raw_docs = list(ingest_plugin.ingest(str(test_dir), kwargs={}))
    print(f"      Ingested {len(raw_docs)} documents")

    # Step 2: Validate
    print("[2/4] Validating documents...")
    valid_docs = validate(raw_docs, ValidationConfig(min_chars=10))
    print(f"      {len(valid_docs)} documents passed validation")

    # Step 3: Chunk
    print("[3/4] Chunking documents...")
    chunker = SimpleChunker(chunk_size=300)
    all_chunks = []
    for doc in valid_docs:
        base_meta = {"source_file": doc.path, "doc_id": Path(doc.path).stem}
        chunks = chunker.chunk_text(doc.content, base_meta)
        all_chunks.extend(chunks)
    print(f"      Created {len(all_chunks)} chunks")

    # Step 4: Embed and store
    print("[4/4] Embedding and storing...")

    # Get embedding plugin
    EmbeddingPlugin = get_llm_plugin(plugin_name="cohere", plugin_type="embedding")
    embedder = EmbeddingPlugin()

    # Get vector DB plugin
    VectorDBPlugin = get_llm_plugin(plugin_name="qdrant", plugin_type="vector_db")
    vector_client = VectorDBPlugin(host="localhost", port=6333)

    # Embed all chunks
    vectors = []
    for chunk in all_chunks:
        vec = embedder.embed(chunk.content)
        vectors.append(vec)

    # Write to vector DB
    writer = VectorDBWriter(client=vector_client)
    writer.upsert(collection=COLLECTION_NAME, chunks=all_chunks, vectors=vectors)
    print(f"      Stored {len(all_chunks)} chunks in '{COLLECTION_NAME}'")

    print("\n✓ Ingestion complete!")


def query_with_rag(query: str) -> None:
    """Run a RAG query against the indexed documents."""
    print("\n" + "=" * 60)
    print("PHASE 2: RAG Query")
    print("=" * 60)
    print(f"\nQuery: {query}")

    # Step 1: Retrieve relevant chunks
    print("\n[1/4] Retrieving relevant chunks...")

    EmbeddingPlugin = get_llm_plugin(plugin_name="cohere", plugin_type="embedding")
    embedder = EmbeddingPlugin()

    VectorDBPlugin = get_llm_plugin(plugin_name="qdrant", plugin_type="vector_db")
    vector_client = VectorDBPlugin(host="localhost", port=6333)

    retriever = DenseRetrievalPlugin(
        client=vector_client,
        embedder=embedder,
        collection=COLLECTION_NAME,
        top_k=5,
    )

    chunks = retriever.retrieve(query)
    print(f"      Retrieved {len(chunks)} chunks")

    # Step 2: Process context
    print("[2/4] Processing context...")
    context_pipeline = ContextPipeline(max_chars=2000)
    processed_chunks = context_pipeline.process(chunks)
    print(f"      Processed to {len(processed_chunks)} chunks")

    # Step 3: Build prompt with RGS
    print("[3/4] Building prompt...")
    rgs = RGS(
        RGSConfig(
            enable_citations=True,
            strict_grounding=True,
            max_chunks=5,
            source_label_prefix="S",
        )
    )
    prompt = rgs.build_prompt(query, processed_chunks)

    # Step 4: Generate answer
    print("[4/4] Generating answer...")

    ChatPlugin = get_llm_plugin(plugin_name="cohere", plugin_type="chat")
    llm = ChatPlugin()

    messages = [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]

    response = llm.chat(messages)
    answer = rgs.build_answer(response, processed_chunks)

    # Display results
    print("\n" + "-" * 60)
    print("ANSWER:")
    print("-" * 60)
    print(answer.answer)

    print("\n" + "-" * 60)
    print("SOURCES:")
    print("-" * 60)
    for source in answer.sources:
        print(f"  [{source.index}] {source.source_id}")
        print(f"      {source.metadata.get('source_file', 'unknown')}")

    print("\n✓ Query complete!")


def main():
    print("=" * 60)
    print("fitz Full Pipeline Demo")
    print("=" * 60)

    # Setup
    test_dir = setup_test_documents()

    # Phase 1: Ingest
    ingest_documents(test_dir)

    # Phase 2: Query
    queries = [
        "What is RAG and how does it work?",
        "What are the different chunking strategies?",
    ]

    for query in queries:
        query_with_rag(query)

    print("\n" + "=" * 60)
    print("Full pipeline demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

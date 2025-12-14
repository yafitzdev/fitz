#!/usr/bin/env python3
"""
fitz Ingestion Example

This example demonstrates the document ingestion flow:
1. Ingest documents from local filesystem
2. Chunk the content
3. Validate documents

Run with:
    # First create some test documents
    mkdir -p test_docs
    echo "RAG systems combine retrieval with generation." > test_docs/pipeline.txt
    echo "Vector databases enable semantic search." > test_docs/vectors.txt

    python examples/ingestion_example.py
"""

from pathlib import Path

from fitz.ingest.chunking.plugins.simple import SimpleChunker
from fitz.ingest.ingestion.registry import get_ingest_plugin
from fitz.ingest.validation.documents import ValidationConfig, validate


def main():
    # Setup: Create test documents if they don't exist
    test_dir = Path("test_docs")
    if not test_dir.exists():
        print("Creating test documents...")
        test_dir.mkdir(exist_ok=True)
        (test_dir / "fitz.pipeline.txt").write_text(
            "RAG (Retrieval-Augmented Generation) systems combine information retrieval "
            "with language model generation. They first retrieve relevant documents from "
            "a knowledge base, then use those documents as context for generating responses. "
            "This approach helps reduce hallucinations and grounds responses in factual data."
        )
        (test_dir / "vectors.txt").write_text(
            "Vector databases store high-dimensional embeddings that represent semantic meaning. "
            "When a query comes in, it's converted to an embedding and compared against stored "
            "vectors using similarity metrics like cosine similarity. Popular vector databases "
            "include Qdrant, Pinecone, Weaviate, and Milvus."
        )
        (test_dir / "empty.txt").write_text("   ")  # Empty file for validation demo
        print(f"Created test documents in {test_dir}/")
    print()

    # Step 1: Ingest documents
    print("=" * 60)
    print("Step 1: Document Ingestion")
    print("=" * 60)

    # Get the local filesystem ingestion plugin
    LocalIngestPlugin = get_ingest_plugin("local")
    ingest_plugin = LocalIngestPlugin()

    # Ingest all documents from the directory
    raw_docs = list(ingest_plugin.ingest(str(test_dir), kwargs={}))

    print(f"Ingested {len(raw_docs)} documents:")
    for doc in raw_docs:
        content_preview = doc.content[:50] + "..." if len(doc.content) > 50 else doc.content
        print(f"  - {doc.path}: {repr(content_preview)}")
    print()

    # Step 2: Validate documents
    print("=" * 60)
    print("Step 2: Document Validation")
    print("=" * 60)

    validation_config = ValidationConfig(
        min_chars=10,
        strip_whitespace=True,
    )

    valid_docs = validate(raw_docs, validation_config)

    print(f"Valid documents: {len(valid_docs)} / {len(raw_docs)}")
    filtered_count = len(raw_docs) - len(valid_docs)
    if filtered_count > 0:
        print(f"  (Filtered out {filtered_count} empty/invalid documents)")
    print()

    # Step 3: Chunk documents
    print("=" * 60)
    print("Step 3: Text Chunking")
    print("=" * 60)

    chunker = SimpleChunker(chunk_size=200)  # Small chunks for demo

    all_chunks = []
    for doc in valid_docs:
        base_meta = {
            "source_file": doc.path,
            "doc_id": Path(doc.path).stem,
        }
        chunks = chunker.chunk_text(doc.content, base_meta)
        all_chunks.extend(chunks)
        print(f"  {doc.path} -> {len(chunks)} chunks")

    print()
    print(f"Total chunks: {len(all_chunks)}")
    print()

    # Show chunk details
    print("Chunk details:")
    print("-" * 40)
    for chunk in all_chunks:
        print(f"ID: {chunk.id}")
        print(f"  doc_id: {chunk.doc_id}")
        print(f"  index: {chunk.chunk_index}")
        print(f"  content: {chunk.content[:80]}...")
        print()

    print("=" * 60)
    print("Ingestion example complete!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Embed chunks using EmbeddingEngine")
    print("  2. Store in vector DB using VectorDBWriter")
    print("  3. Query using RAGPipeline")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
fitz Quickstart Example

This example demonstrates the basic RAG pipeline flow:
1. Create some sample chunks
2. Build a prompt with RGS
3. (Optionally) Call an LLM

Run with:
    python examples/quickstart.py
"""

from fitz.pipeline.context.pipeline import ContextPipeline
from fitz.pipeline.generation.rgs import RGS, RGSConfig


def main():
    # Sample chunks (in real usage, these come from retrieval)
    raw_chunks = [
        {
            "id": "chunk_1",
            "doc_id": "doc_a",
            "chunk_index": 0,
            "content": "fitz is a modular RAG framework designed for production use.",
            "metadata": {"source": "readme.md"},
        },
        {
            "id": "chunk_2",
            "doc_id": "doc_a",
            "chunk_index": 1,
            "content": "It uses a plugin-based architecture where all components are swappable.",
            "metadata": {"source": "readme.md"},
        },
        {
            "id": "chunk_3",
            "doc_id": "doc_b",
            "chunk_index": 0,
            "content": "The RGS module handles prompt construction with citation support.",
            "metadata": {"source": "rgs_docs.md"},
        },
    ]

    # Step 1: Process chunks through context pipeline
    print("=" * 60)
    print("Step 1: Context Processing")
    print("=" * 60)

    context_pipeline = ContextPipeline(max_chars=2000)
    processed_chunks = context_pipeline.process(raw_chunks)

    print(f"Input chunks: {len(raw_chunks)}")
    print(f"Processed chunks: {len(processed_chunks)}")
    for chunk in processed_chunks:
        print(f"  - {chunk.id}: {chunk.content[:50]}...")
    print()

    # Step 2: Build RGS prompt
    print("=" * 60)
    print("Step 2: RGS Prompt Building")
    print("=" * 60)

    rgs = RGS(
        RGSConfig(
            enable_citations=True,
            strict_grounding=True,
            max_chunks=5,
            source_label_prefix="S",
        )
    )

    query = "What is fitz and how does it work?"
    prompt = rgs.build_prompt(query, processed_chunks)

    print("System Prompt:")
    print("-" * 40)
    print(prompt.system)
    print()
    print("User Prompt:")
    print("-" * 40)
    print(prompt.user)
    print()

    # Step 3: Simulate LLM response and build answer
    print("=" * 60)
    print("Step 3: Answer Building")
    print("=" * 60)

    # In real usage, this would come from your LLM
    simulated_llm_response = (
        "fitz is a modular RAG framework designed for production use [S1]. "
        "It features a plugin-based architecture where all components can be swapped [S2]. "
        "The framework includes an RGS module for prompt construction with citation support [S3]."
    )

    answer = rgs.build_answer(simulated_llm_response, processed_chunks)

    print("Answer:")
    print(answer.answer)
    print()
    print("Sources:")
    for source in answer.sources:
        print(f"  [{source.index}] {source.source_id}: {source.metadata}")
    print()

    print("=" * 60)
    print("Quickstart complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

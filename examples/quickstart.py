"""
Quickstart Example - Using the new Fitz Engine API

This example demonstrates the new paradigm-agnostic engine interface.
The code is cleaner and more forward-compatible with future engines (CLaRa, etc.).
"""

from fitz.core import Query, Constraints
from fitz.engines.classic_rag import run_classic_rag, create_classic_rag_engine


# ============================================================================
# OPTION 1: Simple one-off query (easiest)
# ============================================================================

def simple_query():
    """Simple query with default configuration."""
    print("=" * 60)
    print("SIMPLE QUERY")
    print("=" * 60)

    answer = run_classic_rag("What is quantum computing?")

    print(f"\nAnswer: {answer.text}\n")
    print(f"Sources used: {len(answer.provenance)}")
    for i, source in enumerate(answer.provenance, 1):
        print(f"  [{i}] {source.source_id}")


# ============================================================================
# OPTION 2: Query with constraints
# ============================================================================

def query_with_constraints():
    """Query with constraints (filters, source limits)."""
    print("\n" + "=" * 60)
    print("QUERY WITH CONSTRAINTS")
    print("=" * 60)

    # Build constraints
    constraints = Constraints(
        max_sources=5,  # Limit to top 5 sources
        filters={"topic": "quantum_physics"}  # Filter by topic
    )

    answer = run_classic_rag(
        query="Explain quantum entanglement",
        constraints=constraints
    )

    print(f"\nAnswer: {answer.text}\n")
    print(f"Sources (max {constraints.max_sources}): {len(answer.provenance)}")


# ============================================================================
# OPTION 3: Reusable engine (most efficient for multiple queries)
# ============================================================================

def reusable_engine():
    """Create engine once, use for multiple queries."""
    print("\n" + "=" * 60)
    print("REUSABLE ENGINE")
    print("=" * 60)

    # Create engine once
    engine = create_classic_rag_engine("config.yaml")

    # Use for multiple queries (more efficient)
    questions = [
        "What is a qubit?",
        "How does superposition work?",
        "What is quantum decoherence?"
    ]

    for question in questions:
        query = Query(text=question)
        answer = engine.answer(query)
        print(f"\nQ: {question}")
        print(f"A: {answer.text[:100]}...")


# ============================================================================
# OPTION 4: Advanced - with engine-specific hints
# ============================================================================

def advanced_query():
    """Advanced usage with engine-specific metadata."""
    print("\n" + "=" * 60)
    print("ADVANCED QUERY")
    print("=" * 60)

    # Pass engine-specific hints via metadata
    answer = run_classic_rag(
        query="Provide a detailed explanation of quantum tunneling",
        metadata={
            "temperature": 0.3,  # Lower temperature for more focused answers
            "rerank": True,  # Enable reranking
            "model": "claude-3-opus"  # Specific model
        }
    )

    print(f"\nAnswer: {answer.text}\n")
    print("Metadata:")
    print(f"  Engine: {answer.metadata.get('engine')}")
    print(f"  Query: {answer.metadata.get('query_text')}")


# ============================================================================
# OPTION 5: Error handling
# ============================================================================

def error_handling():
    """Demonstrate proper error handling."""
    print("\n" + "=" * 60)
    print("ERROR HANDLING")
    print("=" * 60)

    from fitz.core import QueryError, KnowledgeError, GenerationError

    try:
        # Empty query will raise QueryError
        answer = run_classic_rag("")
    except QueryError as e:
        print(f"Query error: {e}")

    try:
        # Simulate knowledge error (this might not actually fail in practice)
        constraints = Constraints(
            filters={"nonexistent_field": "value"}
        )
        answer = run_classic_rag(
            "Some question",
            constraints=constraints
        )
    except KnowledgeError as e:
        print(f"Knowledge error: {e}")
    except Exception as e:
        print(f"Handled gracefully: {type(e).__name__}")


# ============================================================================
# OPTION 6: Working with the Answer object
# ============================================================================

def working_with_answers():
    """Demonstrate working with Answer objects."""
    print("\n" + "=" * 60)
    print("WORKING WITH ANSWERS")
    print("=" * 60)

    answer = run_classic_rag("What is quantum superposition?")

    # Access answer text
    print(f"\nAnswer text ({len(answer.text)} chars):")
    print(answer.text[:200] + "...")

    # Access provenance
    print(f"\n\nProvenance ({len(answer.provenance)} sources):")
    for prov in answer.provenance:
        print(f"\n  Source: {prov.source_id}")
        if prov.excerpt:
            print(f"  Excerpt: {prov.excerpt[:100]}...")
        if prov.metadata:
            print(f"  Metadata: {list(prov.metadata.keys())}")

    # Access metadata
    print(f"\n\nAnswer metadata:")
    for key, value in answer.metadata.items():
        print(f"  {key}: {value}")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("FITZ QUICKSTART - NEW ENGINE API")
    print("=" * 60)

    try:
        simple_query()
        query_with_constraints()
        reusable_engine()
        advanced_query()
        error_handling()
        working_with_answers()

        print("\n" + "=" * 60)
        print("✅ ALL EXAMPLES COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
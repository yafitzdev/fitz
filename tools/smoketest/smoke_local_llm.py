"""
Consolidated Local LLM Smoketest

Tests all three local LLM components:
1. Chat (generates text)
2. Embedding (creates vectors)
3. Rerank (ranks chunks by relevance)

This smoketest can be run in two ways:

1. As a standalone script (clean error messages):
   python tools/smoketest/smoke_local_llm.py

2. As pytest tests (detailed failure info):
   pytest tools/smoketest/smoke_local_llm.py

"""

import sys
from dataclasses import dataclass
from typing import NoReturn

from fitz_ai.engines.classic_rag.errors.llm import LLMError
from fitz_ai.engines.classic_rag.models.chunk import Chunk
from fitz_ai.llm.chat.plugins.local import LocalChatClient
from fitz_ai.llm.embedding.plugins.local import LocalEmbeddingClient
from fitz_ai.llm.rerank.plugins.local import LocalRerankClient


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    error_message: str | None = None


def local_chat() -> None:
    """Test local chat functionality."""
    print("\n" + "=" * 60)
    print("TEST 1: LOCAL CHAT")
    print("=" * 60)

    llm = LocalChatClient()

    response = llm.chat(
        [
            {"role": "system", "content": "You are a test LLM."},
            {"role": "user", "content": "Say hello in one short sentence."},
        ]
    )

    print(f"✓ Chat response: {response}")


def local_embedding() -> None:
    """Test local embedding functionality."""
    print("\n" + "=" * 60)
    print("TEST 2: LOCAL EMBEDDING")
    print("=" * 60)

    embedder = LocalEmbeddingClient()

    vec = embedder.embed("hello world")

    print(f"✓ Vector length: {len(vec)}")
    print(f"✓ First 10 values: {vec[:10]}")


def local_rerank() -> None:
    """Test local reranking functionality."""
    print("\n" + "=" * 60)
    print("TEST 3: LOCAL RERANK")
    print("=" * 60)

    reranker = LocalRerankClient()

    chunks = [
        Chunk(
            id="c1",
            doc_id="d1",
            chunk_index=0,
            content="The cat sat on the mat.",
            metadata={},
        ),
        Chunk(
            id="c2",
            doc_id="d1",
            chunk_index=1,
            content="Quantum mechanics describes subatomic particles.",
            metadata={},
        ),
        Chunk(
            id="c3",
            doc_id="d1",
            chunk_index=2,
            content="Cats are small domesticated mammals.",
            metadata={},
        ),
    ]

    ranked = reranker.rerank(query="What is a cat?", chunks=chunks)

    print("✓ Reranked chunks:")
    for i, c in enumerate(ranked, start=1):
        print(f"  {i}. {c.id}: {c.content}")


def run_single_test(test_func, test_name: str) -> TestResult:
    """Run a single test and capture its result."""
    try:
        test_func()
        return TestResult(name=test_name, passed=True)
    except LLMError as e:
        # Expected error when Ollama is not running
        print(f"✗ {test_name} failed (Ollama not available)")
        return TestResult(name=test_name, passed=False, error_message=str(e))
    except Exception as e:
        # Unexpected error
        print(f"✗ {test_name} failed with unexpected error: {type(e).__name__}: {e}")
        return TestResult(
            name=test_name, passed=False, error_message=f"Unexpected error: {type(e).__name__}: {e}"
        )


def clean_exit_with_message(message: str) -> NoReturn:
    """Exit cleanly with just the message, no traceback."""
    print("\n" + message, file=sys.stderr)
    sys.exit(1)


def print_summary(results: list[TestResult]) -> None:
    """Print a summary of all test results."""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]

    if passed:
        print(f"\n✓ Passed ({len(passed)}/{len(results)}):")
        for r in passed:
            print(f"  • {r.name}")

    if failed:
        print(f"\n✗ Failed ({len(failed)}/{len(results)}):")
        for r in failed:
            print(f"  • {r.name}")

    print("\n" + "=" * 60)

    if all(r.passed for r in results):
        print("✓ ALL TESTS PASSED")
        print("=" * 60)
        print("\nLocal LLM stack is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 60)
        # Show the error message from the first failure
        first_failure = next((r for r in results if not r.passed), None)
        if first_failure and first_failure.error_message:
            if (
                isinstance(first_failure.error_message, str)
                and "Local LLM fallback" in first_failure.error_message
            ):
                # This is the Ollama setup message - show it
                print(f"\n{first_failure.error_message}")
            else:
                print(f"\nError: {first_failure.error_message}")


def main() -> int:
    """
    Run all local LLM smoketests as a standalone script.

    This provides clean error messages without tracebacks.
    For detailed pytest output, run: pytest tools/smoketest/smoke_local_llm.py
    """
    print("=" * 60)
    print("FITZ LOCAL LLM SMOKETEST")
    print("=" * 60)
    print("\nTesting local LLM components (chat, embedding, rerank)...")
    print("(For detailed pytest output, run: pytest tools/smoketest/smoke_local_llm.py)\n")

    # Run all tests and collect results
    results = []

    try:
        results.append(run_single_test(local_chat, "Local Chat"))
        results.append(run_single_test(local_embedding, "Local Embedding"))
        results.append(run_single_test(local_rerank, "Local Rerank"))

        # Print summary
        print_summary(results)

        # Return success only if all tests passed
        return 0 if all(r.passed for r in results) else 1

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        return 130

    except Exception as e:
        # Completely unexpected error
        print("\n" + "=" * 60)
        print("✗ CRITICAL ERROR")
        print("=" * 60)
        print(f"\n{type(e).__name__}: {e}")
        print("\nThis is an unexpected error. Please report this as a bug.")
        return 1


if __name__ == "__main__":
    # Running as standalone script - use clean error handling
    sys.exit(main())

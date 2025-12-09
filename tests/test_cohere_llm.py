# tests/test_cohere_llm.py
"""
Integration tests for Cohere LLM components:
- Embedding
- Reranking
- Chat

IMPORTANT:
These tests require COHERE_API_KEY to be present in the environment.
If it's missing, the tests will skip automatically.
"""

from __future__ import annotations

import os
import sys

# Import your clients
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient


def require_api_key():
    key = os.getenv("COHERE_API_KEY")
    if not key:
        print("❌ COHERE_API_KEY not set — skipping live tests.")
        return None
    return key


# ---------------------------------------------------------
# Test: Embedding
# ---------------------------------------------------------
def test_embedding():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Embedding ===")

    embedder = CohereEmbeddingClient()

    text = "Hello world from Fitz-RAG!"
    vec = embedder.embed(text)

    print(f"Embedding length: {len(vec)}")
    print(f"First 5 values:   {vec[:5]}")

    assert isinstance(vec, list)
    assert len(vec) > 10  # Cohere embeddings are large vectors

    print("✅ Embedding test passed.")


# ---------------------------------------------------------
# Test: Reranking
# ---------------------------------------------------------
def test_rerank():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Rerank ===")

    reranker = CohereRerankClient()

    query = "What is torque?"
    docs = [
        "Torque is a measure of rotational force.",
        "Bananas are yellow.",
        "Cars use engines to generate torque.",
    ]

    order = reranker.rerank(query, docs, top_n=2)

    print(f"Rerank order: {order}")
    print("Top 2 texts:")
    for idx in order:
        print(f"- {docs[idx]}")

    assert isinstance(order, list)
    assert len(order) == 2
    assert order[0] in range(len(docs))

    print("✅ Rerank test passed.")


# ---------------------------------------------------------
# Test: Chat
# ---------------------------------------------------------
def test_chat():
    if not require_api_key():
        return

    print("\n=== TEST: Cohere Chat ===")

    chat = CohereChatClient()

    response = chat.chat(
        system_prompt="You are a helpful test assistant.",
        user_content="Tell me something about RAG systems."
    )

    print(f"Response: {response}")

    assert isinstance(response, str)
    assert len(response) > 5

    print("✅ Chat test passed.")


# ---------------------------------------------------------
# Runner
# ---------------------------------------------------------
if __name__ == "__main__":
    print("\n🔧 Running Cohere LLM live tests...")
    test_embedding()
    test_rerank()
    test_chat()
    print("\n🎉 All tests finished.")

"""
EasyRAG — zero-configuration preset for fitz-rag.

Example:
    rag = EasyRAG(collection="my_docs", qdrant_host="192.168.178.2")
    rag.ask("What is this about?")

Internally builds:
    - QdrantClient
    - CohereEmbedder
    - RAGRetriever
    - CohereRerankClient
    - CohereChatClient
    - RAGPipeline
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from qdrant_client import QdrantClient

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient


@dataclass
class EasyRAG:
    """
    High-level convenience wrapper around RAGPipeline.

    Users only need:
        rag = EasyRAG(collection="my_collection")
        rag.ask("What is X?")
    """

    collection: str

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    cohere_api_key: Optional[str] = None

    top_k: int = 15
    final_top_k: int = 5
    context_chars: int = 4000
    system_prompt: str = "You are a helpful assistant."

    pipeline: Optional[RAGPipeline] = None

    # ---------------------------------------------------------
    # Build all components automatically
    # ---------------------------------------------------------
    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError(
                "Missing Cohere API key. Set COHERE_API_KEY or pass cohere_api_key=..."
            )

        # Qdrant
        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        # Embedder
        embedder = CohereEmbeddingClient(api_key=key)

        # Retriever
        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            collection=self.collection,
            top_k=self.top_k,
        )

        # Reranker
        reranker = CohereRerankClient(api_key=key)

        # Chat
        chat = CohereChatClient(api_key=key)

        # Pipeline
        self.pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            chat_client=chat,
            system_prompt=self.system_prompt,
            context_chars=self.context_chars,
            final_top_k=self.final_top_k,
        )

    # ---------------------------------------------------------
    # Public API
    # ---------------------------------------------------------
    def ask(self, query: str) -> str:
        """Ask a question with full RAG flow."""
        return self.pipeline.run(query)

    def ask_debug(self, query: str) -> dict:
        """
        Returns {query, context, prompt, answer}.
        Useful for inspecting how RAG made a decision.
        """
        chunks = self.pipeline._retrieve(query)
        reranked = self.pipeline._rerank(query, chunks)
        context = self.pipeline._build_context(reranked)
        prompt = self.pipeline._build_prompt(context, query)
        answer = self.pipeline._chat(prompt)

        return {
            "query": query,
            "context": context,
            "prompt": prompt,
            "answer": answer,
        }

"""
StandardRAG — a configurable but user-friendly RAG preset.

Compared to EasyRAG:
    - exposes all relevant retrieval / rerank / context parameters
    - allows switching embedder and rerank models easily
    - still builds everything automatically for the user
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
class StandardRAG:
    """A flexible RAG preset with sane defaults."""

    collection: str

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # Cohere keys / models
    cohere_api_key: Optional[str] = None
    embed_model: str = "embed-english-v3.0"
    rerank_model: str = "rerank-v3.5"

    # Pipeline parameters
    top_k: int = 20
    final_top_k: int = 5
    context_chars: int = 5000
    system_prompt: str = "You are a helpful assistant."

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass cohere_api_key=...")

        # Qdrant
        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)

        # Embedder (custom model allowed)
        embedder = CohereEmbeddingClient(api_key=key, model=self.embed_model)

        # Retriever
        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            collection=self.collection,
            top_k=self.top_k,
        )

        # Reranker (custom model allowed)
        reranker = CohereRerankClient(api_key=key, model=self.rerank_model)

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

    # User API
    def ask(self, query: str) -> str:
        return self.pipeline.run(query)

    def ask_debug(self, query: str) -> dict:
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

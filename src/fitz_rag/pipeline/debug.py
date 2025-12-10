"""
DebugRAG — a development / introspection pipeline.

Provides detailed breakdown:
    - retrieved chunks
    - reranked order
    - merged context
    - final prompt
    - latencies per stage
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os
import time

from qdrant_client import QdrantClient

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient


@dataclass
class DebugRAG:
    """A developer-focused RAG preset with full introspection."""

    collection: str

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    cohere_api_key: Optional[str] = None

    top_k: int = 20
    final_top_k: int = 5
    context_chars: int = 5000
    system_prompt: str = "You are a helpful assistant."

    pipeline: Optional[RAGPipeline] = None

    def __post_init__(self):
        key = self.cohere_api_key or os.getenv("COHERE_API_KEY")
        if not key:
            raise RuntimeError("Set COHERE_API_KEY or pass cohere_api_key=...")

        qdrant = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        embedder = CohereEmbeddingClient(api_key=key)
        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            collection=self.collection,
            top_k=self.top_k,
        )
        reranker = CohereRerankClient(api_key=key)
        chat = CohereChatClient(api_key=key)

        self.pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            chat_client=chat,
            system_prompt=self.system_prompt,
            context_chars=self.context_chars,
            final_top_k=self.final_top_k,
        )

    # ---------------------------------------------------------
    # DEVELOPER-FACING API
    # ---------------------------------------------------------
    def explain(self, query: str) -> dict:
        times = {}

        t0 = time.time()
        chunks = self.pipeline._retrieve(query)
        times["retrieval_ms"] = int((time.time() - t0) * 1000)

        t1 = time.time()
        reranked = self.pipeline._rerank(query, chunks)
        times["rerank_ms"] = int((time.time() - t1) * 1000)

        t2 = time.time()
        context = self.pipeline._build_context(reranked)
        times["context_ms"] = int((time.time() - t2) * 1000)

        prompt = self.pipeline._build_prompt(context, query)

        t3 = time.time()
        answer = self.pipeline._chat(prompt)
        times["chat_ms"] = int((time.time() - t3) * 1000)

        return {
            "query": query,
            "retrieved": chunks,
            "reranked": reranked,
            "context": context,
            "prompt": prompt,
            "answer": answer,
            "latency": times,
        }

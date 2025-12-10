"""
NoRerankRAG — fastest version of the RAG pipeline.

Skips reranking:
    - Only dense top_k retrieval
    - Useful for low-latency or cheap inference
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import os

from qdrant_client import QdrantClient

from fitz_rag.pipeline.engine import RAGPipeline
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.chat_client import CohereChatClient


@dataclass
class NoRerankRAG:
    """A fast RAG preset without reranking."""

    collection: str

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    cohere_api_key: Optional[str] = None

    top_k: int = 5
    context_chars: int = 4000
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

        # No reranker → replace with a dummy implementation
        from fitz_rag.llm.rerank_client import RerankClient

        class _PassThroughReranker(RerankClient):
            def rerank(self, query, docs, top_n=None):
                # identity order 0,1,2,...
                return list(range(len(docs)))

        reranker = _PassThroughReranker()

        chat = CohereChatClient(api_key=key)

        self.pipeline = RAGPipeline(
            retriever=retriever,
            reranker=reranker,
            chat_client=chat,
            system_prompt=self.system_prompt,
            context_chars=self.context_chars,
            final_top_k=self.top_k,
        )

    def ask(self, query: str) -> str:
        return self.pipeline.run(query)

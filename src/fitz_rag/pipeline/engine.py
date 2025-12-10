"""
RAGPipeline — the core RAG orchestration engine.

This class performs:
  - dense retrieval
  - reranking
  - context building
  - prompt assembly
  - LLM chat

EasyRAG (in easy.py) is a thin wrapper that constructs a configured RAGPipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any

from fitz_rag.retriever.base import BaseRetriever
from fitz_rag.llm.rerank_client import RerankClient
from fitz_rag.llm.chat_client import ChatClient
from fitz_rag.context.builder import build_context


@dataclass
class RAGPipeline:
    """
    The core engine of the fitz-rag retrieval-augmented generation pipeline.

    Users pass in:
      - retriever (BaseRetriever)
      - reranker (RerankClient)
      - chat_client (ChatClient)

    Example:
        pipeline = RAGPipeline(retriever, reranker, chat_client)
        answer = pipeline.run("What is this?")
    """

    retriever: BaseRetriever
    reranker: RerankClient
    chat_client: ChatClient

    system_prompt: str = "You are a helpful assistant."
    context_chars: int = 4000
    final_top_k: int = 5

    # ---------------------------------------------------------
    # STEP 1 — Dense Retrieval
    # ---------------------------------------------------------
    def _retrieve(self, query: str) -> List[Dict[str, Any]]:
        results = self.retriever.retrieve(query)
        if not results:
            return []

        chunks = []
        for r in results:
            chunks.append(
                {
                    "text": r.text,
                    "file": r.metadata.get("file", "unknown"),
                }
            )
        return chunks

    # ---------------------------------------------------------
    # STEP 2 — Reranking
    # ---------------------------------------------------------
    def _rerank(self, query: str, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chunks:
            return []

        texts = [c["text"] for c in chunks]
        indices = self.reranker.rerank(query, texts, top_n=self.final_top_k)
        return [chunks[i] for i in indices]

    # ---------------------------------------------------------
    # STEP 3 — Build Context
    # ---------------------------------------------------------
    def _build_context(self, chunks: List[Dict[str, Any]]) -> str:
        return build_context(chunks, max_chars=self.context_chars)

    # ---------------------------------------------------------
    # STEP 4 — Build Prompt
    # ---------------------------------------------------------
    def _build_prompt(self, context: str, query: str) -> str:
        return f"""
CONTEXT:
{context}

QUESTION:
{query}

INSTRUCTIONS:
- Answer ONLY using the context above.
- If the context does not contain the answer, say: "I don't know."
""".strip()

    # ---------------------------------------------------------
    # STEP 5 — Chat with LLM
    # ---------------------------------------------------------
    def _chat(self, prompt: str) -> str:
        return self.chat_client.chat(
            system_prompt=self.system_prompt,
            user_content=prompt,
        )

    # ---------------------------------------------------------
    # PUBLIC ENTRYPOINT
    # ---------------------------------------------------------
    def run(self, query: str) -> str:
        chunks = self._retrieve(query)
        if not chunks:
            return "No relevant information found."

        reranked = self._rerank(query, chunks)
        context = self._build_context(reranked)
        prompt = self._build_prompt(context, query)

        answer = self._chat(prompt)
        return answer

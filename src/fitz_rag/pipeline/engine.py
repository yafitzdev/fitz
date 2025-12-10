from __future__ import annotations

from typing import List, Optional

from qdrant_client import QdrantClient

from fitz_rag.models.chunk import Chunk
from fitz_rag.generation.rgs import RGS, RGSAnswer, RGSConfig as RGSRuntimeConfig
from fitz_rag.retriever.dense_retriever import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank_client import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient
from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config


class RAGPipeline:
    """
    Final v0.1.0 RAG pipeline:
        retriever → chunks
        RGS → prompt
        LLM → answer
        RGS → structured answer
    """

    def __init__(self, retriever, llm, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

    def run(self, query: str) -> RGSAnswer:
        """
        Execute a full RAG cycle:
            1) retrieve chunks
            2) generate RGS prompt
            3) call LLM
            4) wrap answer with sources
        """
        # 1. Retrieve (retriever may internally embed + rerank)
        chunks: List[Chunk] = self.retriever.retrieve(query)

        # 2. Build prompt using RGS
        prompt = self.rgs.build_prompt(query, chunks)

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3. LLM call
        raw = self.llm.chat(messages)

        # 4. Structured answer
        return self.rgs.build_answer(raw, chunks)

    # ------------------------------------------------------------------
    # Unified config entry-point
    # ------------------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        """
        Build a full RAGPipeline from a unified RAGConfig.

        Assumes:
          - Qdrant as vector DB
          - Cohere for embeddings, rerank, and chat (as per config)
        """

        # --------- Resolve API key (LLM / embedding / rerank share it) ----------
        key: Optional[str] = (
            cfg.llm.api_key
            or cfg.embedding.api_key
            or cfg.rerank.api_key
        )

        if key is None:
            raise RuntimeError(
                "No API key found in configuration (llm.api_key / embedding.api_key / rerank.api_key)."
            )

        # --------- Vector DB (Qdrant) ----------
        qdrant = QdrantClient(
            host=cfg.retriever.qdrant_host,
            port=cfg.retriever.qdrant_port,
        )

        # --------- Embeddings ----------
        if cfg.embedding.provider.lower() != "cohere":
            raise ValueError(f"Unsupported embedding provider: {cfg.embedding.provider!r}")
        embedder = CohereEmbeddingClient(api_key=key, model=cfg.embedding.model)

        # --------- Reranker (optional) ----------
        if cfg.rerank.enabled:
            if not cfg.rerank.provider or cfg.rerank.provider.lower() != "cohere":
                raise ValueError(f"Unsupported rerank provider: {cfg.rerank.provider!r}")
            if not cfg.rerank.model:
                raise ValueError("rerank.model must be set if rerank.enabled = true")
            reranker = CohereRerankClient(api_key=key, model=cfg.rerank.model)
        else:
            reranker = None

        # --------- Chat LLM ----------
        if cfg.llm.provider.lower() != "cohere":
            raise ValueError(f"Unsupported llm provider: {cfg.llm.provider!r}")
        chat = CohereChatClient(
            api_key=key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
        )

        # --------- Retriever ----------
        retriever = RAGRetriever(
            client=qdrant,
            embedder=embedder,
            reranker=reranker,
            collection=cfg.retriever.collection,
            top_k=cfg.retriever.top_k,
        )

        # --------- RGS (convert from RAGConfig.rgs → runtime RGSConfig) ----------
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
        )
        rgs = RGS(config=rgs_cfg)

        return cls(retriever=retriever, llm=chat, rgs=rgs)


def create_pipeline(
    name: str,
    config: Optional[RAGConfig] = None,
    **kwargs,
) -> RAGPipeline:
    """
    Factory for building pipelines.

    Two main usage modes:

      1) Config-based (preferred):
            cfg = RAGConfig.from_dict(load_config())
            pipe = create_pipeline("standard", config=cfg)

         In this mode, `name` is currently ignored, and the unified config
         fully defines the stack. (Future versions may use `name` to select
         presets.)

      2) Legacy preset wrappers (Easy/Standard/Fast/Debug):
            pipe = create_pipeline("easy", collection="my_docs")

         This uses the existing EasyRAG / FastRAG / StandardRAG / DebugRAG
         builder classes and returns their internal RAGPipeline instance.
    """
    # 1) Config-based construction (unified YAML → Pydantic → pipeline)
    if config is not None:
        return RAGPipeline.from_config(config)

    # 2) Preset-based construction (legacy-style wrappers)
    name = name.lower()

    if name in ("easy", "simple"):
        from fitz_rag.pipeline.easy import EasyRAG
        return EasyRAG(**kwargs).pipeline

    if name in ("fast",):
        from fitz_rag.pipeline.fast import FastRAG
        return FastRAG(**kwargs).pipeline

    if name in ("standard", "default"):
        from fitz_rag.pipeline.standard import StandardRAG
        return StandardRAG(**kwargs).pipeline

    if name in ("debug", "verbose"):
        from fitz_rag.pipeline.debug import DebugRAG
        return DebugRAG(**kwargs).pipeline

    raise ValueError(f"Unknown pipeline type: {name!r}")


def create_pipeline_from_yaml(user_config_path: Optional[str] = None) -> RAGPipeline:
    """
    Convenience helper:
        - loads merged config from default.yaml + optional user config
        - validates into RAGConfig
        - builds a RAGPipeline from it
    """
    raw = load_config(user_config_path=user_config_path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

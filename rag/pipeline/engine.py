# fitz_rag/pipeline/engine.py
from __future__ import annotations

from typing import Optional

from rag.config.schema import RAGConfig
from rag.config.loader import load_config

from rag.retrieval.plugins.dense import RAGRetriever

from core.llm.chat import ChatEngine
from core.vector_db import get_vector_db_plugin

from rag.generation.rgs import (
    RGS,
    RGSConfig as RGSRuntimeConfig,
    RGSAnswer,
)

from rag.exceptions.pipeline import PipelineError, RGSGenerationError
from rag.exceptions.llm import LLMError

from core.logging.logger import get_logger
from core.logging.tags import PIPELINE, VECTOR_DB

logger = get_logger(__name__)


class RAGPipeline:
    """
    Final clean RAG pipeline.
    Architecture:
        vector_db → retrieval → rgs → llm → final answer
    """

    def __init__(self, retriever: RAGRetriever, llm: ChatEngine, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    # ---------------------------------------------------------
    # Main run
    # ---------------------------------------------------------
    def run(self, query: str) -> RGSAnswer:
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # 1) Retrieve
        try:
            chunks = self.retriever.retrieve(query)
        except Exception as e:
            logger.error(f"{PIPELINE} Retriever failed: {e}")
            raise PipelineError("Retriever failed") from e

        # 2) Build RGS prompt
        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {e}")
            raise RGSGenerationError("Failed to build RGS prompt") from e

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3) LLM call
        try:
            raw = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"{PIPELINE} LLM chat failed: {e}")
            raise LLMError("LLM chat operation failed") from e

        # 4) Build RGS answer
        try:
            answer = self.rgs.build_answer(raw, chunks)
            logger.info(f"{PIPELINE} Pipeline run completed")
            return answer
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {e}")
            raise RGSGenerationError("Failed to build RGS answer") from e

    # ---------------------------------------------------------
    # Factory from config
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # ---------------- Vector DB client ----------------
        VectorDBCls = get_vector_db_plugin("qdrant")
        vector_client = VectorDBCls(
            host=cfg.retriever.qdrant_host,
            port=cfg.retriever.qdrant_port,
        )
        logger.info(
            f"{VECTOR_DB} Using vector DB plugin='qdrant', "
            f"collection='{cfg.retriever.collection}'"
        )

        # ---------------- Chat Engine ------------------
        chat_engine = ChatEngine.from_name(
            cfg.llm.provider.lower(),
            api_key=cfg.llm.api_key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
        )

        # ---------------- Retriever --------------------
        retriever = RAGRetriever(
            client=vector_client,
            embed_cfg=cfg.embedding,
            retriever_cfg=cfg.retriever,
            rerank_cfg=cfg.rerank,
        )

        # ---------------- RGS Config -------------------
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
            include_query_in_context=cfg.rgs.include_query_in_context,
            source_label_prefix=getattr(cfg.rgs, "source_label_prefix", "S"),
        )
        rgs = RGS(config=rgs_cfg)

        logger.info(f"{PIPELINE} RAGPipeline successfully created")
        return cls(retriever=retriever, llm=chat_engine, rgs=rgs)


# ---------------------------------------------------------
# Factory for YAML-driven construction
# ---------------------------------------------------------
def create_pipeline_from_yaml(path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Loading config from YAML")
    raw = load_config(path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

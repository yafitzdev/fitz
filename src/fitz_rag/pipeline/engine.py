from __future__ import annotations

from typing import Optional

from qdrant_client import QdrantClient

from fitz_rag.config.schema import RAGConfig
from fitz_rag.config.loader import load_config

from fitz_rag.retriever.plugins.dense import RAGRetriever
from fitz_rag.llm.embedding_client import CohereEmbeddingClient
from fitz_rag.llm.rerank.plugins.cohere import CohereRerankClient
from fitz_rag.llm.chat_client import CohereChatClient

from fitz_rag.generation.rgs import RGS, RGSConfig as RGSRuntimeConfig
from fitz_rag.generation.rgs import RGSAnswer

from fitz_rag.exceptions.pipeline import PipelineError, RGSGenerationError
from fitz_rag.exceptions.llm import LLMError

from fitz_stack.logging import get_logger
from fitz_stack.logging_tags import PIPELINE

logger = get_logger(__name__)


class RAGPipeline:
    """
    Final v0.1.0 RAG pipeline using ONLY the unified config system.

    Pipeline:
        1. Retriever retrieves + reranks chunks
        2. RGS builds prompt
        3. LLM executes
        4. RGS builds answer structure
    """

    def __init__(self, retriever, llm, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    # ---------------------------------------------------------
    # Main RAG execution
    # ---------------------------------------------------------
    def run(self, query: str) -> RGSAnswer:
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # 1) RETRIEVE
        logger.debug(f"{PIPELINE} Retrieving chunks")
        try:
            chunks = self.retriever.retrieve(query)
        except Exception as e:
            logger.error(f"{PIPELINE} Retriever failed: {e}")
            raise PipelineError("Retriever failed") from e

        # 2) BUILD RGS PROMPT
        logger.debug(f"{PIPELINE} Building RGS prompt")
        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {e}")
            raise RGSGenerationError("Failed to build RGS prompt") from e

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        # 3) LLM CALL
        logger.debug(f"{PIPELINE} Calling LLM")
        try:
            raw = self.llm.chat(messages)
        except Exception as e:
            logger.error(f"{PIPELINE} LLM chat operation failed: {e}")
            raise LLMError("LLM chat operation failed") from e

        # 4) STRUCTURE ANSWER
        logger.debug(f"{PIPELINE} Building structured answer")
        try:
            answer = self.rgs.build_answer(raw, chunks)
            logger.info(f"{PIPELINE} Pipeline run completed successfully")
            return answer
        except Exception as e:
            logger.error(f"{PIPELINE} Failed to build RGS answer: {e}")
            raise RGSGenerationError("Failed to build RGS answer") from e

    # ---------------------------------------------------------
    # Factory from unified config
    # ---------------------------------------------------------
    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        logger.info(f"{PIPELINE} Building RAGPipeline from unified config")

        # Resolve API key (shared for embedding/chat/rerank)
        key = (
            cfg.llm.api_key
            or cfg.embedding.api_key
            or cfg.rerank.api_key
        )
        if key is None:
            raise PipelineError("No API key provided in any config section.")

        # ---------------- Qdrant Client ----------------
        logger.debug(f"{PIPELINE} Connecting to Qdrant at {cfg.retriever.qdrant_host}:{cfg.retriever.qdrant_port}")
        qdrant = QdrantClient(
            host=cfg.retriever.qdrant_host,
            port=cfg.retriever.qdrant_port,
        )

        # ---------------- Embedder ---------------------
        if cfg.embedding.provider.lower() != "cohere":
            raise PipelineError(f"Unsupported embedding provider: {cfg.embedding.provider}")

        logger.debug(f"{PIPELINE} Initializing embedding model '{cfg.embedding.model}'")
        embedder = CohereEmbeddingClient(
            api_key=key,
            model=cfg.embedding.model,
        )

        # ---------------- Reranker (optional) ---------
        if cfg.rerank.enabled:
            if cfg.rerank.provider.lower() != "cohere":
                raise PipelineError(f"Unsupported rerank provider: {cfg.rerank.provider}")
            if not cfg.rerank.model:
                raise PipelineError("rerank.model must be set if reranking is enabled")

            logger.debug(f"{PIPELINE} Initializing rerank model '{cfg.rerank.model}'")
            reranker = CohereRerankClient(
                api_key=key,
                model=cfg.rerank.model,
            )
        else:
            logger.debug(f"{PIPELINE} Reranker disabled")
            reranker = None

        # ---------------- Chat LLM ---------------------
        if cfg.llm.provider.lower() != "cohere":
            raise PipelineError(f"Unsupported LLM provider: {cfg.llm.provider}")

        logger.debug(f"{PIPELINE} Initializing chat model '{cfg.llm.model}'")
        chat = CohereChatClient(
            api_key=key,
            model=cfg.llm.model,
            temperature=cfg.llm.temperature,
        )

        # ---------------- Retriever --------------------
        logger.debug(f"{PIPELINE} Constructing retriever")
        retriever = RAGRetriever(
            client=qdrant,
            embed_cfg=cfg.embedding,
            retriever_cfg=cfg.retriever,
            rerank_cfg=cfg.rerank,
        )

        # override embedder + reranker
        retriever.embedder = embedder
        retriever.reranker = reranker

        # ---------------- RGS --------------------------
        logger.debug(f"{PIPELINE} Initializing RGS runtime config")
        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
        )
        rgs = RGS(config=rgs_cfg)

        logger.info(f"{PIPELINE} RAGPipeline successfully constructed")
        return cls(retriever=retriever, llm=chat, rgs=rgs)


# ---------------------------------------------------------
# Pipeline factories
# ---------------------------------------------------------
def create_pipeline(name: str, config: Optional[RAGConfig] = None, **kwargs) -> RAGPipeline:
    logger.debug(f"{PIPELINE} create_pipeline requested: '{name}'")

    if config is not None:
        logger.debug(f"{PIPELINE} Creating pipeline from unified config")
        return RAGPipeline.from_config(config)

    name = name.lower()

    if name in ("easy", "simple"):
        logger.debug(f"{PIPELINE} Delegating to EasyRAG")
        from fitz_rag.pipeline.easy import EasyRAG
        return EasyRAG(**kwargs).pipeline

    if name in ("fast",):
        logger.debug(f"{PIPELINE} Delegating to FastRAG")
        from fitz_rag.pipeline.fast import FastRAG
        return FastRAG(**kwargs).pipeline

    if name in ("standard", "default"):
        logger.debug(f"{PIPELINE} Delegating to StandardRAG")
        from fitz_rag.pipeline.standard import StandardRAG
        return StandardRAG(**kwargs).pipeline

    if name in ("debug", "verbose"):
        logger.debug(f"{PIPELINE} Delegating to DebugRAG")
        from fitz_rag.pipeline.debug import DebugRAG
        return DebugRAG(**kwargs).pipeline

    raise PipelineError(f"Unknown pipeline type: {name}")


def create_pipeline_from_yaml(user_config_path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Creating pipeline from YAML config")
    raw = load_config(user_config_path=user_config_path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

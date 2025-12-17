# pipeline/pipeline/engine.py
from __future__ import annotations

from typing import Optional

from fitz.engines.classic_rag.errors.llm import LLMError
from fitz.llm.chat import ChatEngine
from fitz.llm.embedding.engine import EmbeddingEngine
from fitz.llm.registry import resolve_llm_plugin
from fitz.llm.rerank.engine import RerankEngine
from fitz.logging.logger import get_logger
from fitz.logging.tags import PIPELINE, VECTOR_DB
from fitz.vector_db.registry import get_vector_db_plugin
fitz.engines.classic_rag.generation.retrieval_guided.synthesis import RGS, RGSAnswer
fitz.engines.classic_rag.generation.retrieval_guided.synthesis import RGSConfig as RGSRuntimeConfig
from fitz.engines.classic_rag.pipeline.config.loader import load_config
from fitz.engines.classic_rag.pipeline.config.schema import RAGConfig
from fitz.engines.classic_rag.pipeline.context.pipeline import ContextPipeline
from fitz.engines.classic_rag.pipeline.exceptions.pipeline import PipelineError, RGSGenerationError
from fitz.engines.classic_rag.retrieval.runtime.engine import RetrieverEngine

logger = get_logger(__name__)


class RAGPipeline:
    """
    vector_db → retrieval → context-processing → rgs → llm → final answer
    """

    def __init__(
        self,
        retriever: RetrieverEngine,
        llm: ChatEngine,
        rgs: RGS,
        context: ContextPipeline | None = None,
    ):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs
        self.context = context or ContextPipeline()

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    def run(self, query: str) -> RGSAnswer:
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        try:
            raw_chunks = self.retriever.retrieve(query)
        except Exception as exc:
            logger.error(f"{PIPELINE} Retriever failed: {exc}")
            raise PipelineError("Retriever failed") from exc

        try:
            chunks = self.context.process(raw_chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Context processing failed: {exc}")
            raise PipelineError("Context processing failed") from exc

        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {exc}")
            raise RGSGenerationError("Failed to build RGS prompt") from exc

        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        try:
            raw = self.llm.chat(messages)
        except Exception as exc:
            logger.error(f"{PIPELINE} LLM chat failed: {exc}")
            raise LLMError("LLM chat operation failed") from exc

        try:
            answer = self.rgs.build_answer(raw, chunks)
            logger.info(f"{PIPELINE} Pipeline run completed")
            return answer
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {exc}")
            raise RGSGenerationError("Failed to build RGS answer") from exc

    @classmethod
    def from_config(cls, cfg: RAGConfig) -> "RAGPipeline":
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        VectorDBCls = get_vector_db_plugin(cfg.vector_db.plugin_name)
        vector_client = VectorDBCls(**cfg.vector_db.kwargs)
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{cfg.vector_db.plugin_name}'")

        # Chat (local-first fallback)
        ChatCls = resolve_llm_plugin(
            plugin_type="chat",
            requested_name=cfg.llm.plugin_name,
        )
        chat_plugin = ChatCls(**cfg.llm.kwargs)
        chat_engine = ChatEngine(chat_plugin)

        # Embedding (local-first fallback)
        EmbedCls = resolve_llm_plugin(
            plugin_type="embedding",
            requested_name=cfg.embedding.plugin_name,
        )
        embed_plugin = EmbedCls(**cfg.embedding.kwargs)
        embedder = EmbeddingEngine(embed_plugin)

        # Rerank (local-first fallback)
        rerank_engine: RerankEngine | None = None
        if cfg.rerank.enabled:
            RerankCls = resolve_llm_plugin(
                plugin_type="rerank",
                requested_name=cfg.rerank.plugin_name,  # type: ignore[arg-type]
            )
            rerank_plugin = RerankCls(**cfg.rerank.kwargs)
            rerank_engine = RerankEngine(rerank_plugin)

        retriever = RetrieverEngine.from_name(
            cfg.retriever.plugin_name,
            client=vector_client,
            retriever_cfg=cfg.retriever,
            embedder=embedder,
            rerank_engine=rerank_engine,
        )

        rgs_cfg = RGSRuntimeConfig(
            enable_citations=cfg.rgs.enable_citations,
            strict_grounding=cfg.rgs.strict_grounding,
            answer_style=cfg.rgs.answer_style,
            max_chunks=cfg.rgs.max_chunks,
            max_answer_chars=cfg.rgs.max_answer_chars,
            include_query_in_context=cfg.rgs.include_query_in_context,
            source_label_prefix=cfg.rgs.source_label_prefix,
        )
        rgs = RGS(config=rgs_cfg)

        logger.info(f"{PIPELINE} RAGPipeline successfully created")
        return cls(retriever=retriever, llm=chat_engine, rgs=rgs, context=ContextPipeline())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGPipeline":
        """
        Create a RAGPipeline from a configuration dictionary.

        This method is useful for creating pipelines from presets or
        dictionaries without manually constructing RAGConfig objects.

        Args:
            config_dict: Configuration dictionary matching RAGConfig schema

        Returns:
            RAGPipeline instance

        Example:
            >>> from fitz.engines.classic_rag.config.presets import get_preset
            >>> config = get_preset("local")
            >>> pipeline = RAGPipeline.from_dict(config)
        """
        logger.info(f"{PIPELINE} Creating RAGPipeline from dict")
        cfg = RAGConfig.from_dict(config_dict)
        return cls.from_config(cfg)


def create_pipeline_from_yaml(path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Loading config from YAML")
    raw = load_config(path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

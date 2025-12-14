# pipeline/pipeline/engine.py
from __future__ import annotations

from typing import Optional

from fitz.core.exceptions.llm import LLMError
from fitz.core.llm.chat import ChatEngine
from fitz.core.llm.embedding.engine import EmbeddingEngine
from fitz.core.llm.registry import get_llm_plugin
from fitz.core.llm.rerank.engine import RerankEngine
from fitz.core.logging.logger import get_logger
from fitz.core.logging.tags import PIPELINE, VECTOR_DB
from fitz.core.vector_db.registry import get_vector_db_plugin
from fitz.pipeline.config.loader import load_config
from fitz.pipeline.config.schema import RAGConfig
from fitz.pipeline.context.pipeline import ContextPipeline
from fitz.pipeline.exceptions.pipeline import PipelineError, RGSGenerationError
from fitz.pipeline.generation.rgs import RGS, RGSAnswer
from fitz.pipeline.generation.rgs import RGSConfig as RGSRuntimeConfig
from fitz.pipeline.retrieval import RetrieverEngine

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

        chat_engine = ChatEngine.from_name(cfg.llm.plugin_name, **cfg.llm.kwargs)

        EmbedCls = get_llm_plugin(plugin_name=cfg.embedding.plugin_name, plugin_type="embedding")
        embed_plugin = EmbedCls(**cfg.embedding.kwargs)
        embedder = EmbeddingEngine(embed_plugin)

        rerank_engine: RerankEngine | None = None
        if cfg.rerank.enabled:
            RerankCls = get_llm_plugin(
                plugin_name=cfg.rerank.plugin_name,  # type: ignore[arg-type]
                plugin_type="rerank",
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


def create_pipeline_from_yaml(path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Loading config from YAML")
    raw = load_config(path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

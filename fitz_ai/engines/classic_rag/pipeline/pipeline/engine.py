# fitz/engines/classic_rag/pipeline/pipeline/engine.py
"""
RAGPipeline - Core orchestration for retrieval-augmented generation.

Flow: vector_db → retrieval → context-processing → rgs → llm → final answer
"""
from __future__ import annotations


from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config
from fitz_ai.engines.classic_rag.exceptions import (
    LLMError,
    PipelineError,
    RGSGenerationError,
)
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSAnswer,
)
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGSConfig as RGSRuntimeConfig,
)
from fitz_ai.engines.classic_rag.pipeline.context.pipeline import ContextPipeline
from fitz_ai.engines.classic_rag.retrieval.runtime.engine import RetrieverEngine
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE, VECTOR_DB
from fitz_ai.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)


class RAGPipeline:
    """
    RAG Pipeline orchestrator.

    Flow: vector_db → retrieval → context-processing → rgs → llm → final answer
    """

    def __init__(
        self,
        retriever: RetrieverEngine,
        chat,
        rgs: RGS,
        context: ContextPipeline | None = None,
    ):
        self.retriever = retriever
        self.chat = chat
        self.rgs = rgs
        self.context = context or ContextPipeline()

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    def run(self, query: str) -> RGSAnswer:
        """Execute the RAG pipeline for a query."""
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # Step 1: Retrieve relevant chunks
        try:
            raw_chunks = self.retriever.retrieve(query)
        except Exception as exc:
            logger.error(f"{PIPELINE} Retriever failed: {exc}")
            raise PipelineError("Retriever failed") from exc

        # Step 2: Process context (dedupe, group, merge, pack)
        try:
            chunks = self.context.process(raw_chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Context processing failed: {exc}")
            raise PipelineError("Context processing failed") from exc

        # Step 3: Build RGS prompt
        try:
            prompt = self.rgs.build_prompt(query, chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {exc}")
            raise RGSGenerationError("Failed to build RGS prompt") from exc

        # Step 4: Generate answer via LLM
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        try:
            raw = self.chat.chat(messages)
        except Exception as exc:
            logger.error(f"{PIPELINE} LLM chat failed: {exc}")
            raise LLMError("LLM chat operation failed") from exc

        # Step 5: Structure the answer
        try:
            answer = self.rgs.build_answer(raw, chunks)
            logger.info(f"{PIPELINE} Pipeline run completed")
            return answer
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {exc}")
            raise RGSGenerationError("Failed to build RGS answer") from exc

    @classmethod
    def from_config(cls, cfg: ClassicRagConfig) -> "RAGPipeline":
        """Create a RAGPipeline from configuration."""
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # Vector DB
        vector_client = get_vector_db_plugin(cfg.vector_db.plugin_name, **cfg.vector_db.kwargs)
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{cfg.vector_db.plugin_name}'")

        # Chat LLM
        chat_plugin = get_llm_plugin(
            plugin_type="chat", plugin_name=cfg.chat.plugin_name, **cfg.chat.kwargs
        )
        logger.info(f"{PIPELINE} Using chat plugin='{cfg.chat.plugin_name}'")

        # Embedding
        embedder = get_llm_plugin(
            plugin_type="embedding", plugin_name=cfg.embedding.plugin_name, **cfg.embedding.kwargs
        )
        logger.info(f"{PIPELINE} Using embedding plugin='{cfg.embedding.plugin_name}'")

        # Rerank (optional)
        rerank_plugin = None
        if cfg.rerank.enabled:
            rerank_plugin = get_llm_plugin(
                plugin_type="rerank", plugin_name=cfg.rerank.plugin_name, **cfg.rerank.kwargs
            )
            logger.info(f"{PIPELINE} Using rerank plugin='{cfg.rerank.plugin_name}'")

        # Retriever
        retriever = RetrieverEngine.from_name(
            cfg.retriever.plugin_name,
            client=vector_client,
            retriever_cfg=cfg.retriever,
            embedder=embedder,
            rerank_engine=rerank_plugin,
        )

        # RGS
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
        return cls(retriever=retriever, chat=chat_plugin, rgs=rgs, context=ContextPipeline())

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGPipeline":
        """Create a RAGPipeline from a configuration dictionary."""
        cfg = ClassicRagConfig.model_validate(config_dict)
        return cls.from_config(cfg)

    @classmethod
    def from_yaml(cls, config_path: str) -> "RAGPipeline":
        """Create a RAGPipeline from a YAML configuration file."""
        cfg = load_config(config_path)
        return cls.from_config(cfg)


def create_pipeline_from_yaml(path: str | None = None) -> RAGPipeline:
    """Create a RAGPipeline from a YAML config file."""
    cfg = load_config(path)
    return RAGPipeline.from_config(cfg)

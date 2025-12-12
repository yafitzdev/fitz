# rag/pipeline/engine.py
from __future__ import annotations

from typing import Any, Optional

from rag.config.loader import load_config
from rag.config.schema import RAGConfig
from rag.exceptions.llm import LLMError
from rag.exceptions.pipeline import PipelineError, RGSGenerationError
from rag.generation.rgs import RGS, RGSAnswer, RGSConfig as RGSRuntimeConfig
from rag.retrieval.engine import RetrieverEngine

from core.llm.chat import ChatEngine
from core.llm.embedding.engine import EmbeddingEngine
from core.llm.registry import get_llm_plugin
from core.llm.rerank.engine import RerankEngine
from core.logging.logger import get_logger
from core.logging.tags import PIPELINE, VECTOR_DB
from core.vector_db import get_vector_db_plugin

logger = get_logger(__name__)


def _cfg_to_kwargs(cfg: Any, *, exclude: set[str] | None = None) -> dict[str, Any]:
    if cfg is None:
        return {}

    exclude = exclude or set()

    if hasattr(cfg, "model_dump"):
        data = cfg.model_dump(exclude_none=True)
    elif hasattr(cfg, "dict"):
        data = cfg.dict(exclude_none=True)
    elif isinstance(cfg, dict):
        data = {k: v for k, v in cfg.items() if v is not None}
    else:
        data = {k: v for k, v in vars(cfg).items() if not k.startswith("_") and v is not None}

    for k in exclude:
        data.pop(k, None)

    data.pop("plugin_name", None)
    data.pop("provider", None)
    return data


def _select_plugin_name(cfg: Any, *, label: str) -> str:
    name = getattr(cfg, "plugin_name", None) or getattr(cfg, "provider", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"{label}.plugin_name (or {label}.provider) must be set")
    return name


class RAGPipeline:
    """
    Final clean RAG pipeline.
    Architecture:
        vector_db → retrieval → rgs → llm → final answer
    """

    def __init__(self, retriever: RetrieverEngine, llm: ChatEngine, rgs: RGS):
        self.retriever = retriever
        self.llm = llm
        self.rgs = rgs

        logger.info(f"{PIPELINE} RAGPipeline initialized")

    def run(self, query: str) -> RGSAnswer:
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        try:
            chunks = self.retriever.retrieve(query)
        except Exception as exc:
            logger.error(f"{PIPELINE} Retriever failed: {exc}")
            raise PipelineError("Retriever failed") from exc

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

        # ---------------- Vector DB client ----------------
        if not hasattr(cfg, "vector_db"):
            raise ValueError("RAGConfig.vector_db must be provided (provider-specific fields must not live in retriever config)")

        vector_db_name = _select_plugin_name(cfg.vector_db, label="vector_db")
        VectorDBCls = get_vector_db_plugin(vector_db_name)
        vector_client = VectorDBCls(**_cfg_to_kwargs(cfg.vector_db))
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{vector_db_name}'")

        # ---------------- Chat Engine ------------------
        chat_name = _select_plugin_name(cfg.llm, label="llm")
        chat_engine = ChatEngine.from_name(chat_name, **_cfg_to_kwargs(cfg.llm))

        # ---------------- Embedder ----------------
        embed_name = _select_plugin_name(cfg.embedding, label="embedding")
        EmbedCls = get_llm_plugin(plugin_name=embed_name, plugin_type="embedding")
        embed_plugin = EmbedCls(**_cfg_to_kwargs(cfg.embedding))
        embedder = EmbeddingEngine(embed_plugin)

        # ---------------- Optional rerank ----------------
        rerank_engine: RerankEngine | None = None
        if getattr(cfg.rerank, "enabled", False):
            rerank_name = _select_plugin_name(cfg.rerank, label="rerank")
            RerankCls = get_llm_plugin(plugin_name=rerank_name, plugin_type="rerank")
            rerank_plugin = RerankCls(**_cfg_to_kwargs(cfg.rerank, exclude={"enabled"}))
            rerank_engine = RerankEngine(rerank_plugin)

        # ---------------- Retriever --------------------
        retriever_name = _select_plugin_name(cfg.retriever, label="retriever")
        retriever = RetrieverEngine.from_name(
            retriever_name,
            client=vector_client,
            retriever_cfg=cfg.retriever,
            embedder=embedder,
            rerank_engine=rerank_engine,
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


def create_pipeline_from_yaml(path: Optional[str] = None) -> RAGPipeline:
    logger.debug(f"{PIPELINE} Loading config from YAML")
    raw = load_config(path)
    cfg = RAGConfig.from_dict(raw)
    return RAGPipeline.from_config(cfg)

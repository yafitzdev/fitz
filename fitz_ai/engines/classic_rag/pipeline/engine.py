# fitz_ai/engines/classic_rag/pipeline/engine.py
"""
RAGPipeline - Core orchestration for retrieval-augmented generation.

Flow: retrieval → constraints → answer_mode → context-processing → rgs → llm → answer
"""

from __future__ import annotations

from typing import Sequence

from fitz_ai.core.answer_mode import AnswerMode
from fitz_ai.core.answer_mode_resolver import resolve_answer_mode
from fitz_ai.engines.classic_rag.config import ClassicRagConfig, load_config
from fitz_ai.engines.classic_rag.constraints import (
    ConstraintPlugin,
    ConstraintResult,
    get_default_constraints,
)
from fitz_ai.engines.classic_rag.exceptions import (
    LLMError,
    PipelineError,
    RGSGenerationError,
)
from fitz_ai.engines.classic_rag.generation.answer_mode.instructions import (
    get_mode_instruction,
)
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGS,
    RGSAnswer,
)
from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
    RGSConfig as RGSRuntimeConfig,
)
from fitz_ai.engines.classic_rag.pipeline.pipeline import ContextPipeline
from fitz_ai.engines.classic_rag.retrieval.registry import get_retrieval_plugin
from fitz_ai.llm.registry import get_llm_plugin
from fitz_ai.logging.logger import get_logger
from fitz_ai.logging.tags import PIPELINE, VECTOR_DB
from fitz_ai.vector_db.registry import get_vector_db_plugin

logger = get_logger(__name__)


# =============================================================================
# RAGPipeline
# =============================================================================


class RAGPipeline:
    """
    RAG Pipeline orchestrator.

    Flow: retrieval → constraints → context-processing → rgs → llm → answer
    """

    def __init__(
        self,
        retrieval,  # RetrievalPipelineFromYaml
        chat,
        rgs: RGS,
        context: ContextPipeline | None = None,
        constraints: Sequence[ConstraintPlugin] | None = None,
    ):
        self.retrieval = retrieval
        self.chat = chat
        self.rgs = rgs
        self.context = context or ContextPipeline()

        # Default constraints: ConflictAware + InsufficientEvidence
        # Users can override by passing constraints=[] to disable
        if constraints is None:
            self.constraints: list[ConstraintPlugin] = get_default_constraints()
        else:
            self.constraints = list(constraints)

        logger.info(
            f"{PIPELINE} RAGPipeline initialized with {retrieval.plugin_name} retrieval, "
            f"{len(self.constraints)} constraint(s)"
        )

    def run(self, query: str) -> RGSAnswer:
        """Execute the RAG pipeline for a query."""
        logger.info(f"{PIPELINE} Running pipeline for query='{query[:50]}...'")

        # Step 1: Retrieve relevant chunks (using YAML-based plugin)
        try:
            raw_chunks = self.retrieval.retrieve(query)
        except Exception as exc:
            logger.error(f"{PIPELINE} Retrieval failed: {exc}")
            raise PipelineError("Retrieval failed") from exc

        # Step 2: Apply constraints
        constraint_results = self._apply_all_constraints(query, raw_chunks)

        # Step 3: Resolve answer mode from constraint signals
        answer_mode = resolve_answer_mode(constraint_results)
        logger.info(f"{PIPELINE} Answer mode resolved: {answer_mode.value}")

        # Step 4: Process context (dedupe, group, merge, pack)
        try:
            chunks = self.context.process(raw_chunks)
        except Exception as exc:
            logger.error(f"{PIPELINE} Context processing failed: {exc}")
            raise PipelineError("Context processing failed") from exc

        # Step 5: Build RGS prompt with answer mode instruction
        try:
            prompt = self.rgs.build_prompt(query, chunks)
            prompt = self._apply_answer_mode_to_prompt(prompt, answer_mode)
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to build RGS prompt: {exc}")
            raise RGSGenerationError("Failed to build RGS prompt") from exc

        # Step 6: Generate answer via LLM
        messages = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": prompt.user},
        ]

        try:
            raw = self.chat.chat(messages)
        except Exception as exc:
            logger.error(f"{PIPELINE} LLM chat failed: {exc}")
            raise LLMError("LLM chat operation failed") from exc

        # Step 7: Structure the answer with mode
        try:
            answer = self.rgs.build_answer(raw, chunks, mode=answer_mode)
            logger.info(f"{PIPELINE} Pipeline run completed (mode={answer_mode.value})")
            return answer
        except Exception as exc:
            logger.error(f"{PIPELINE} Failed to structure RGS answer: {exc}")
            raise RGSGenerationError("Failed to build RGS answer") from exc

    def _apply_all_constraints(
        self,
        query: str,
        chunks,
    ) -> list[ConstraintResult]:
        """Apply all constraints and return individual results."""
        results: list[ConstraintResult] = []

        for constraint in self.constraints:
            try:
                result = constraint.apply(query, chunks)
                results.append(result)
            except Exception as e:
                logger.warning(
                    f"{PIPELINE} Constraint '{constraint.name}' raised exception: {e}"
                )
                # Fail-safe: continue without this constraint's result
                continue

        return results

    def _apply_answer_mode_to_prompt(self, prompt, answer_mode: AnswerMode):
        """Prepend answer mode instruction to system prompt."""
        from fitz_ai.engines.classic_rag.generation.retrieval_guided.synthesis import (
            RGSPrompt,
        )

        instruction = get_mode_instruction(answer_mode)
        modified_system = f"{instruction}\n\n{prompt.system}"

        return RGSPrompt(
            system=modified_system,
            user=prompt.user,
        )

    @classmethod
    def from_config(
        cls,
        cfg: ClassicRagConfig,
        constraints: Sequence[ConstraintPlugin] | None = None,
    ) -> "RAGPipeline":
        """Create a RAGPipeline from configuration."""
        logger.info(f"{PIPELINE} Constructing RAGPipeline from config")

        # Vector DB
        vector_client = get_vector_db_plugin(
            cfg.vector_db.plugin_name, **cfg.vector_db.kwargs
        )
        logger.info(f"{VECTOR_DB} Using vector DB plugin='{cfg.vector_db.plugin_name}'")

        # Chat LLM
        chat_plugin = get_llm_plugin(
            plugin_type="chat", plugin_name=cfg.chat.plugin_name, **cfg.chat.kwargs
        )
        logger.info(f"{PIPELINE} Using chat plugin='{cfg.chat.plugin_name}'")

        # Embedding
        embedder = get_llm_plugin(
            plugin_type="embedding",
            plugin_name=cfg.embedding.plugin_name,
            **cfg.embedding.kwargs,
        )
        logger.info(f"{PIPELINE} Using embedding plugin='{cfg.embedding.plugin_name}'")

        # Rerank (optional)
        reranker = None
        if cfg.rerank.enabled and cfg.rerank.plugin_name:
            reranker = get_llm_plugin(
                plugin_type="rerank",
                plugin_name=cfg.rerank.plugin_name,
                **cfg.rerank.kwargs,
            )
            logger.info(f"{PIPELINE} Using rerank plugin='{cfg.rerank.plugin_name}'")

        # Retrieval (YAML-based plugin)
        retrieval = get_retrieval_plugin(
            plugin_name=cfg.retrieval.plugin_name,
            vector_client=vector_client,
            embedder=embedder,
            collection=cfg.retrieval.collection,
            reranker=reranker,
            top_k=cfg.retrieval.top_k,
        )
        logger.info(f"{PIPELINE} Using retrieval plugin='{cfg.retrieval.plugin_name}'")

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
        return cls(
            retrieval=retrieval,
            chat=chat_plugin,
            rgs=rgs,
            context=ContextPipeline(),
            constraints=constraints,
        )

    @classmethod
    def from_dict(cls, config_dict: dict) -> "RAGPipeline":
        """Create a RAGPipeline from a configuration dictionary."""
        cfg = ClassicRagConfig.from_dict(config_dict)
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

# fitz_ai/engines/clara/engine.py
"""
ClaraEngine - Knowledge engine implementation for Apple's CLaRa paradigm.

This engine wraps Apple's CLaRa (Continuous Latent Reasoning) model behind
the paradigm-agnostic KnowledgeEngine interface.

CLaRa differs from Classic RAG:
- Documents are compressed into continuous memory tokens (16x-128x compression)
- Retrieval happens in latent space via cosine similarity
- Retriever and generator are jointly optimized (end-to-end)
- Single language modeling loss trains both retrieval and generation

Requirements:
    pip install torch transformers accelerate bitsandbytes peft
    # Or: pip install fitz-ai[clara]
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from fitz_ai.core import (
    Answer,
    ConfigurationError,
    GenerationError,
    KnowledgeError,
    Provenance,
    Query,
    QueryError,
)
from fitz_ai.engines.clara.config.schema import ClaraConfig

logger = logging.getLogger(__name__)


def _ensure_clara_model_available(model_path: str) -> Path:
    """
    Ensure CLaRa model files are available locally.

    Downloads from HuggingFace if needed and returns path to local files.
    """
    from huggingface_hub import snapshot_download

    # Check if it's already a local path
    local_path = Path(model_path)
    if local_path.exists() and (local_path / "modeling_clara.py").exists():
        return local_path

    # Download from HuggingFace
    # CLaRa models have subdirectories like compression-16, compression-128
    repo_id = "apple/CLaRa-7B-Instruct"  # Default repo
    subfolder = "compression-16"  # Default compression

    # Parse model path to extract repo and subfolder
    if "/" in model_path:
        parts = model_path.split("/")
        if len(parts) >= 3 and parts[0] == "apple":
            repo_id = f"{parts[0]}/{parts[1]}"
            if len(parts) >= 3:
                subfolder = parts[2]

    logger.info(f"Downloading CLaRa model from {repo_id}/{subfolder}...")

    # Create cache directory
    cache_dir = Path.home() / ".cache" / "fitz" / "clara"
    cache_dir.mkdir(parents=True, exist_ok=True)

    local_dir = cache_dir / repo_id.replace("/", "_") / subfolder

    # Download model files
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    snapshot_download(
        repo_id=repo_id,
        allow_patterns=[f"{subfolder}/*"],
        local_dir=str(local_dir.parent.parent),
    )

    return local_dir.parent / subfolder


class ClaraEngine:
    """
    CLaRa (Continuous Latent Reasoning) engine implementation.

    This engine implements the CLaRa paradigm using Apple's official model:
    1. Compress documents into continuous memory tokens (offline)
    2. Embed query into same latent space
    3. Retrieve via cosine similarity in latent space
    4. Generate answer from query + top-k compressed docs

    Key advantages over Classic RAG:
    - 16x-128x document compression while preserving semantics
    - Unified retrieval-generation optimization
    - No separate embedding model needed
    - Better multi-hop reasoning

    Examples:
        >>> from fitz_ai.engines.clara import ClaraEngine
        >>> from fitz_ai.engines.clara.config.schema import ClaraConfig, ClaraModelConfig
        >>>
        >>> # Use 4-bit quantization for lower VRAM usage
        >>> config = ClaraConfig(
        ...     model=ClaraModelConfig(load_in_4bit=True)
        ... )
        >>> engine = ClaraEngine(config)
        >>>
        >>> # Add documents to knowledge base
        >>> engine.add_documents(["Doc 1 content...", "Doc 2 content..."])
        >>>
        >>> # Query
        >>> query = Query(text="What is quantum computing?")
        >>> answer = engine.answer(query)
        >>> print(answer.text)
    """

    def __init__(self, config: ClaraConfig):
        """
        Initialize the CLaRa engine.

        Args:
            config: ClaraConfig object with model, compression, retrieval settings

        Raises:
            ConfigurationError: If configuration is invalid or model cannot be loaded
        """
        self._config = config
        self._model = None
        self._doc_texts: List[str] = []  # Original document texts
        self._model_path: Optional[Path] = None

        try:
            self._initialize_model()
        except ImportError as e:
            raise ConfigurationError(
                "CLaRa requires additional dependencies. "
                "Install with: pip install fitz-ai[clara] "
                f"or pip install torch transformers accelerate bitsandbytes peft. Error: {e}"
            ) from e
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize CLaRa engine: {e}") from e

    def _initialize_model(self) -> None:
        """
        Load the CLaRa model from HuggingFace or local cache.

        CLaRa uses a custom model class that must be loaded via its own
        from_pretrained method, not HuggingFace's AutoModel.
        """
        import torch

        model_config = self._config.model

        # Ensure model files are available
        self._model_path = _ensure_clara_model_available(model_config.model_name_or_path)

        # Add model path to Python path for custom module loading
        model_parent = str(self._model_path)
        if model_parent not in sys.path:
            sys.path.insert(0, model_parent)

        # Import CLaRa's custom model class
        from modeling_clara import CLaRa

        logger.info(f"Loading CLaRa model from: {self._model_path}")

        # Determine quantization setting
        quantization = "no"
        if model_config.load_in_4bit:
            quantization = "int4"
        elif model_config.load_in_8bit:
            quantization = "int8"

        # Load model with Apple's from_pretrained
        self._model = CLaRa.from_pretrained(
            str(self._model_path),
            quantization=quantization,
            device_map="auto",
        )

        self._model.eval()

        # Log memory usage
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"CLaRa model loaded. GPU memory: {mem_gb:.2f} GB")
        else:
            logger.info("CLaRa model loaded on CPU")

    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add documents to the knowledge base.

        Documents are stored for later compression during queries.
        CLaRa compresses documents on-the-fly during generation.

        Args:
            documents: List of document texts to add
            doc_ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            List of document IDs

        Raises:
            KnowledgeError: If documents cannot be added
        """
        if doc_ids is None:
            import uuid

            doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]

        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")

        try:
            # Store documents for later use
            for doc_text in documents:
                self._doc_texts.append(doc_text)

            logger.info(f"Added {len(documents)} documents to CLaRa knowledge base")
            return doc_ids

        except Exception as e:
            raise KnowledgeError(f"Failed to add documents: {e}") from e

    def answer(self, query: Query) -> Answer:
        """
        Execute a query against knowledge and return an answer.

        CLaRa's answer generation:
        1. Format documents for CLaRa's input format
        2. Compress documents and generate answer in one pass
        3. Return answer with provenance

        Args:
            query: Query object containing the question text

        Returns:
            Answer object with the answer text and source provenance

        Raises:
            QueryError: If the query is invalid
            KnowledgeError: If retrieval fails
            GenerationError: If answer generation fails
        """
        if not query.text.strip():
            raise QueryError("Query text cannot be empty")

        if not self._doc_texts:
            raise KnowledgeError("No documents in knowledge base. Call add_documents() first.")

        try:
            # Determine how many docs to use
            top_k = min(
                self._config.retrieval.top_k,
                len(self._doc_texts),
            )
            if query.constraints and query.constraints.max_sources:
                top_k = min(top_k, query.constraints.max_sources)

            # Format documents for CLaRa (list of lists)
            # CLaRa expects: [[doc1, doc2, ...]] for a single question
            documents = [self._doc_texts[:top_k]]
            questions = [query.text]

            # Set generation top_k
            self._model.generation_top_k = top_k

            # Generate answer using CLaRa's generate method
            # Use generate_from_text for instruct variant (stage1_2)
            gen_config = self._config.generation
            answers = self._model.generate_from_text(
                questions=questions,
                documents=documents,
                max_new_tokens=gen_config.max_new_tokens,
            )

            answer_text = answers[0] if answers else ""

            # Build provenance from used documents
            provenance = []
            for i, doc_text in enumerate(self._doc_texts[:top_k]):
                provenance.append(
                    Provenance(
                        source_id=f"doc_{i}",
                        excerpt=(doc_text[:200] + "..." if len(doc_text) > 200 else doc_text),
                        metadata={
                            "rank": i + 1,
                            "compression_rate": self._config.compression.compression_rate,
                        },
                    )
                )

            return Answer(
                text=answer_text,
                provenance=provenance,
                metadata={
                    "engine": "clara",
                    "variant": self._config.model.variant,
                    "compression_rate": self._config.compression.compression_rate,
                    "num_docs_used": top_k,
                },
            )

        except QueryError:
            raise
        except KnowledgeError:
            raise
        except Exception as e:
            raise GenerationError(f"CLaRa generation failed: {e}") from e

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "num_documents": len(self._doc_texts),
            "compression_rate": self._config.compression.compression_rate,
            "model_variant": self._config.model.variant,
            "quantization": "4-bit"
            if self._config.model.load_in_4bit
            else ("8-bit" if self._config.model.load_in_8bit else "none"),
        }

    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        """
        self._doc_texts.clear()
        logger.info("CLaRa knowledge base cleared")

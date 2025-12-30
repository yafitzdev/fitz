# fitz_ai/engines/clara/engine.py
"""
ClaraEngine - Apple's CLaRa (Continuous Latent Reasoning) implementation.

This engine implements the CLaRa paradigm from Apple Research:
1. Documents are pre-compressed into continuous memory tokens at ingestion time
2. Compressed representations are stored for fast retrieval
3. Retrieval happens via cosine similarity in the latent space
4. Generation uses pre-compressed documents (compress once, query many)

Hardware Requirements:
    - VRAM: 16GB+ recommended (RTX 4090, A100, etc.)
    - 4-bit quantization reduces to ~12GB minimum
    - NOT suitable for consumer GPUs with 8GB VRAM

Software Requirements:
    pip install torch transformers accelerate bitsandbytes peft
    # Or: pip install fitz-ai[clara]

Note: CLaRa is a research model optimized for quality over speed.
For consumer hardware, use Classic RAG instead.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

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


def _ensure_clara_model_available(model_path: str, quiet: bool = True) -> Path:
    """
    Ensure CLaRa model files are available locally.

    Downloads from HuggingFace if needed and returns path to local files.
    """
    # Suppress huggingface_hub progress bars
    if quiet:
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

    from huggingface_hub import snapshot_download

    # Check if it's already a local path
    local_path = Path(model_path)
    if local_path.exists() and (local_path / "modeling_clara.py").exists():
        return local_path

    # Download from HuggingFace
    repo_id = "apple/CLaRa-7B-Instruct"
    subfolder = "compression-16"

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
    CLaRa (Continuous Latent Reasoning) engine.

    Implements Apple's CLaRa paradigm - a unified model for compression,
    retrieval, and generation in a shared latent space.

    Architecture:
        1. Ingestion: Documents compressed into memory tokens (16x-128x compression)
        2. Retrieval: Cosine similarity in latent space (no separate embedding model)
        3. Generation: Uses pre-compressed representations (compress once, query many)

    Requirements:
        - VRAM: 16GB+ (12GB minimum with 4-bit quantization)
        - Model: apple/CLaRa-7B-Instruct (7B parameters)

    When to use Clara vs Classic RAG:
        - Clara: Research/quality focus, high-end GPU available
        - Classic RAG: Production use, consumer hardware, speed priority

    Examples:
        >>> from fitz_ai.engines.clara import ClaraEngine
        >>> from fitz_ai.engines.clara.config.schema import ClaraConfig
        >>>
        >>> engine = ClaraEngine(ClaraConfig())
        >>> engine.add_documents(["Doc 1...", "Doc 2..."])  # Compressed once
        >>> answer = engine.answer(Query(text="Question?"))  # Fast - no recompression
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
        self._model_path: Optional[Path] = None

        # Document storage
        self._doc_texts: List[str] = []  # Original texts (for provenance)
        self._doc_ids: List[str] = []  # Document IDs
        self._compressed_docs: Optional[torch.Tensor] = (
            None  # [num_docs, num_mem_tokens, hidden_dim]
        )
        self._doc_embeddings: Optional[torch.Tensor] = None  # [num_docs, hidden_dim] for retrieval

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
        """Load the CLaRa model with quantization."""
        import warnings

        # Suppress progress bars and warnings
        _original_env = {
            "TQDM_DISABLE": os.environ.get("TQDM_DISABLE"),
            "HF_HUB_DISABLE_PROGRESS_BARS": os.environ.get("HF_HUB_DISABLE_PROGRESS_BARS"),
            "TRANSFORMERS_NO_ADVISORY_WARNINGS": os.environ.get(
                "TRANSFORMERS_NO_ADVISORY_WARNINGS"
            ),
            "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        }
        os.environ["TQDM_DISABLE"] = "1"
        os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
        os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        # Suppress verbose loggers
        _loggers_to_quiet = [
            "transformers",
            "transformers.modeling_utils",
            "accelerate",
            "peft",
            "huggingface_hub",
            "modeling_clara",
            "torch",
            "bitsandbytes",
        ]
        _original_levels = {}
        for name in _loggers_to_quiet:
            _logger = logging.getLogger(name)
            _original_levels[name] = _logger.level
            _logger.setLevel(logging.CRITICAL)

        # Redirect stdout/stderr during model load
        import io

        _orig_stdout, _orig_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = io.StringIO(), io.StringIO()

        try:
            model_config = self._config.model
            self._model_path = _ensure_clara_model_available(model_config.model_name_or_path)

            # Add model path for custom module loading
            model_parent = str(self._model_path)
            if model_parent not in sys.path:
                sys.path.insert(0, model_parent)

            from modeling_clara import CLaRa

            # Determine quantization (4-bit by default now)
            quantization = "no"
            if model_config.load_in_4bit:
                quantization = "int4"
            elif model_config.load_in_8bit:
                quantization = "int8"

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                self._model = CLaRa.from_pretrained(
                    str(self._model_path),
                    quantization=quantization,
                    device_map="auto",
                )

            self._model.eval()

        finally:
            sys.stdout, sys.stderr = _orig_stdout, _orig_stderr
            for key, val in _original_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val
            for name, level in _original_levels.items():
                logging.getLogger(name).setLevel(level)

        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"CLaRa model loaded (4-bit). GPU memory: {mem_gb:.2f} GB")
        else:
            logger.info("CLaRa model loaded on CPU")

    def _compress_document(self, text: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress a single document into latent representation.

        Returns:
            Tuple of (compressed_tokens, retrieval_embedding)
            - compressed_tokens: [1, num_mem_tokens, hidden_dim]
            - retrieval_embedding: [1, hidden_dim] mean-pooled for retrieval
        """
        # Use model's compress_documents if available
        if hasattr(self._model, "compress_documents"):
            result = self._model.compress_documents([text])
            # compress_documents returns (compressed_tensor, loss) tuple
            if isinstance(result, tuple):
                compressed = result[0]
            else:
                compressed = result
            # compressed shape: [1, num_mem_tokens, hidden_dim]
            retrieval_emb = compressed.mean(dim=1)  # Mean pool for retrieval
            return compressed, retrieval_emb

        # Fallback: manual compression using model internals
        # This path is for when compress_documents isn't exposed
        # CLaRa stores tokenizer as decoder_tokenizer
        tokenizer = getattr(self._model, "decoder_tokenizer", None) or self._model.tokenizer
        device = next(self._model.parameters()).device

        # Tokenize with encoder markers
        enc_text = f"<ENC>{text}"
        inputs = tokenizer(
            enc_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._config.compression.doc_max_length,
            padding=True,
        ).to(device)

        with torch.no_grad():
            # Get hidden states from the model's encoder path
            outputs = self._model.decoder(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True,
            )
            hidden = outputs.hidden_states[-1]  # Last layer

            # Mean pool for retrieval embedding
            mask = inputs["attention_mask"].unsqueeze(-1)
            retrieval_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        return hidden, retrieval_emb

    def add_documents(self, documents: List[str], doc_ids: Optional[List[str]] = None) -> List[str]:
        """
        Add and PRE-COMPRESS documents into the knowledge base.

        This is where the actual compression happens - documents are compressed
        into latent representations and stored for fast retrieval later.

        Args:
            documents: List of document texts to add
            doc_ids: Optional list of document IDs

        Returns:
            List of document IDs

        Raises:
            KnowledgeError: If compression fails
        """
        if doc_ids is None:
            import uuid

            start_idx = len(self._doc_texts)
            doc_ids = [f"doc_{start_idx + i}_{uuid.uuid4().hex[:8]}" for i in range(len(documents))]

        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")

        try:
            batch_size = self._config.compression.compression_batch_size

            all_compressed = []
            all_embeddings = []

            # Compress documents in batches (no gradients needed for inference)
            with torch.no_grad():
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i : i + batch_size]

                    if hasattr(self._model, "compress_documents"):
                        result = self._model.compress_documents(batch_docs)
                        # compress_documents returns (compressed_tensor, loss) tuple
                        compressed = result[0] if isinstance(result, tuple) else result
                        # compressed: [batch, num_mem_tokens, hidden_dim]
                        retrieval_embs = compressed.mean(dim=1)  # [batch, hidden_dim]
                    else:
                        # Fallback: compress individually
                        batch_compressed = []
                        batch_embs = []
                        for doc in batch_docs:
                            comp, emb = self._compress_document(doc)
                            batch_compressed.append(comp)
                            batch_embs.append(emb)
                        compressed = torch.cat(batch_compressed, dim=0)
                        retrieval_embs = torch.cat(batch_embs, dim=0)

                    all_compressed.append(compressed.detach())
                    all_embeddings.append(retrieval_embs.detach())

            # Concatenate with existing documents
            new_compressed = torch.cat(all_compressed, dim=0)
            new_embeddings = torch.cat(all_embeddings, dim=0)

            if self._compressed_docs is None:
                self._compressed_docs = new_compressed
                self._doc_embeddings = new_embeddings
            else:
                self._compressed_docs = torch.cat([self._compressed_docs, new_compressed], dim=0)
                self._doc_embeddings = torch.cat([self._doc_embeddings, new_embeddings], dim=0)

            # Store texts and IDs for provenance
            self._doc_texts.extend(documents)
            self._doc_ids.extend(doc_ids)

            logger.info(
                f"Compressed and added {len(documents)} documents. "
                f"Total: {len(self._doc_texts)} docs, "
                f"Compressed shape: {self._compressed_docs.shape}"
            )
            return doc_ids

        except Exception as e:
            raise KnowledgeError(f"Failed to compress documents: {e}") from e

    def _retrieve_top_k(self, query: str, top_k: int) -> Tuple[List[int], torch.Tensor]:
        """
        Retrieve top-k documents via cosine similarity in latent space.

        Args:
            query: Query text
            top_k: Number of documents to retrieve

        Returns:
            Tuple of (indices, similarities)
        """
        if self._doc_embeddings is None or len(self._doc_embeddings) == 0:
            raise KnowledgeError("No documents in knowledge base")

        device = self._doc_embeddings.device

        # Encode query into same latent space
        with torch.no_grad():
            if hasattr(self._model, "encode_query"):
                query_emb = self._model.encode_query(query)
            else:
                # Fallback: use decoder hidden states for query embedding
                tokenizer = getattr(self._model, "decoder_tokenizer", None) or self._model.tokenizer
                inputs = tokenizer(
                    query,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True,
                ).to(device)

                outputs = self._model.decoder(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    output_hidden_states=True,
                )
                hidden = outputs.hidden_states[-1]
                mask = inputs["attention_mask"].unsqueeze(-1)
                query_emb = (hidden * mask).sum(dim=1) / mask.sum(dim=1)

        # Normalize for cosine similarity
        query_emb = F.normalize(query_emb.to(device), p=2, dim=-1)
        doc_embs = F.normalize(self._doc_embeddings, p=2, dim=-1)

        # Compute similarities
        similarities = torch.matmul(query_emb, doc_embs.T).squeeze(0)

        # Get top-k
        top_k = min(top_k, len(self._doc_texts))
        top_similarities, top_indices = torch.topk(similarities, top_k)

        return top_indices.tolist(), top_similarities

    def answer(self, query: Query) -> Answer:
        """
        Answer a query using latent retrieval and compressed generation.

        Flow:
        1. Encode query into latent space
        2. Retrieve top-k documents via cosine similarity
        3. Generate answer from pre-compressed representations

        Args:
            query: Query object

        Returns:
            Answer with text and provenance

        Raises:
            QueryError: If query is invalid
            KnowledgeError: If retrieval fails
            GenerationError: If generation fails
        """
        if not query.text.strip():
            raise QueryError("Query text cannot be empty")

        if self._compressed_docs is None or len(self._doc_texts) == 0:
            raise KnowledgeError("No documents in knowledge base. Call add_documents() first.")

        try:
            # Determine top-k
            top_k = min(self._config.retrieval.top_k, len(self._doc_texts))
            if query.constraints and query.constraints.max_sources:
                top_k = min(top_k, query.constraints.max_sources)

            # Retrieve via latent space similarity
            retrieved_indices, similarities = self._retrieve_top_k(query.text, top_k)

            # Get pre-compressed representations for retrieved docs
            retrieved_compressed = self._compressed_docs[retrieved_indices]
            # Shape: [num_retrieved, num_mem_tokens, hidden_dim]

            gen_config = self._config.generation

            # Try to use pre-compressed generation (compress once, generate many)
            if hasattr(self._model, "generate_from_compressed_documents_and_questions"):
                try:
                    # Set generation_top_k to match retrieved docs
                    self._model.generation_top_k = len(retrieved_indices)

                    logger.info(
                        f"Using pre-compressed generation with {len(retrieved_indices)} docs"
                    )
                    with torch.no_grad():
                        answers = self._model.generate_from_compressed_documents_and_questions(
                            questions=[query.text],
                            compressed_documents=retrieved_compressed,  # 3D tensor, no unsqueeze
                            max_new_tokens=gen_config.max_new_tokens,
                        )
                    answer_text = answers[0] if answers else ""
                    logger.info("Pre-compressed generation succeeded")
                except Exception as e:
                    # Fallback to generate_from_text if compressed generation fails
                    logger.warning(
                        f"Compressed generation failed ({e}), falling back to generate_from_text"
                    )
                    retrieved_texts = [self._doc_texts[i] for i in retrieved_indices]
                    with torch.no_grad():
                        answers = self._model.generate_from_text(
                            questions=[query.text],
                            documents=[retrieved_texts],
                            max_new_tokens=gen_config.max_new_tokens,
                        )
                    answer_text = answers[0] if answers else ""
            else:
                # Fallback: use generate_from_text
                retrieved_texts = [self._doc_texts[i] for i in retrieved_indices]
                with torch.no_grad():
                    answers = self._model.generate_from_text(
                        questions=[query.text],
                        documents=[retrieved_texts],
                        max_new_tokens=gen_config.max_new_tokens,
                    )
                answer_text = answers[0] if answers else ""

            # Build provenance with retrieval scores
            provenance = []
            for rank, (idx, sim) in enumerate(zip(retrieved_indices, similarities.tolist())):
                doc_text = self._doc_texts[idx]
                provenance.append(
                    Provenance(
                        source_id=self._doc_ids[idx],
                        excerpt=(doc_text[:200] + "..." if len(doc_text) > 200 else doc_text),
                        metadata={
                            "rank": rank + 1,
                            "similarity_score": round(sim, 4),
                            "compression_rate": self._config.compression.compression_rate,
                            "original_index": idx,
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
                    "num_docs_retrieved": len(retrieved_indices),
                    "retrieval_method": "cosine_similarity_latent",
                    "quantization": "4-bit"
                    if self._config.model.load_in_4bit
                    else ("8-bit" if self._config.model.load_in_8bit else "none"),
                },
            )

        except (QueryError, KnowledgeError):
            raise
        except Exception as e:
            raise GenerationError(f"CLaRa generation failed: {e}") from e

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        stats = {
            "num_documents": len(self._doc_texts),
            "compression_rate": self._config.compression.compression_rate,
            "model_variant": self._config.model.variant,
            "quantization": "4-bit"
            if self._config.model.load_in_4bit
            else ("8-bit" if self._config.model.load_in_8bit else "none"),
        }

        if self._compressed_docs is not None:
            stats["compressed_shape"] = list(self._compressed_docs.shape)
            stats["memory_tokens_per_doc"] = self._compressed_docs.shape[1]
            stats["hidden_dim"] = self._compressed_docs.shape[2]
            # Estimate memory usage
            mem_bytes = self._compressed_docs.element_size() * self._compressed_docs.nelement()
            stats["compressed_memory_mb"] = round(mem_bytes / (1024 * 1024), 2)

        return stats

    def clear_knowledge_base(self) -> None:
        """Clear all documents and compressed representations."""
        self._doc_texts.clear()
        self._doc_ids.clear()
        self._compressed_docs = None
        self._doc_embeddings = None
        logger.info("CLaRa knowledge base cleared")

    def ingest(self, source: Path, collection: str) -> Dict[str, Any]:
        """
        Ingest documents from source, compress, and save to persistent storage.

        Args:
            source: Path to documents (file or directory)
            collection: Collection name for storage

        Returns:
            Dict with ingestion stats
        """
        import json

        from fitz_ai.core.paths import FitzPaths
        from fitz_ai.ingestion.reader.engine import IngestionEngine
        from fitz_ai.ingestion.reader.registry import get_ingest_plugin

        # Read documents
        IngestPluginCls = get_ingest_plugin("local")
        ingest_plugin = IngestPluginCls()
        ingest_engine = IngestionEngine(plugin=ingest_plugin, kwargs={})
        raw_docs = list(ingest_engine.run(str(source)))

        if not raw_docs:
            raise KnowledgeError(f"No documents found in {source}")

        # Add documents (this compresses them)
        doc_texts = [doc.content for doc in raw_docs]
        doc_ids = [str(doc.path) for doc in raw_docs]
        self.add_documents(doc_texts, doc_ids=doc_ids)

        # Save to persistent storage
        storage_dir = FitzPaths.ensure_clara_storage(collection)

        # Save metadata as JSON
        metadata = {
            "doc_texts": self._doc_texts,
            "doc_ids": self._doc_ids,
            "compression_rate": self._config.compression.compression_rate,
            "model_variant": self._config.model.variant,
        }
        metadata_path = storage_dir / "metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2))

        # Save tensors
        torch.save(self._compressed_docs, storage_dir / "compressed_docs.pt")
        torch.save(self._doc_embeddings, storage_dir / "doc_embeddings.pt")

        logger.info(f"CLaRa collection '{collection}' saved to {storage_dir}")

        return {
            "documents": len(raw_docs),
            "compressed_shape": list(self._compressed_docs.shape),
            "memory_tokens_per_doc": self._compressed_docs.shape[1],
            "storage_path": str(storage_dir),
        }

    def load(self, collection: str) -> None:
        """
        Load compressed documents from persistent storage.

        Args:
            collection: Collection name to load
        """
        import json

        from fitz_ai.core.paths import FitzPaths

        storage_dir = FitzPaths.clara_storage(collection)
        if not storage_dir.exists():
            raise KnowledgeError(f"Collection '{collection}' not found at {storage_dir}")

        # Load metadata
        metadata_path = storage_dir / "metadata.json"
        if not metadata_path.exists():
            raise KnowledgeError(f"Metadata file not found in {storage_dir}")

        metadata = json.loads(metadata_path.read_text())
        self._doc_texts = metadata["doc_texts"]
        self._doc_ids = metadata["doc_ids"]

        # Load tensors
        device = next(self._model.parameters()).device
        self._compressed_docs = torch.load(
            storage_dir / "compressed_docs.pt", map_location=device, weights_only=True
        )
        self._doc_embeddings = torch.load(
            storage_dir / "doc_embeddings.pt", map_location=device, weights_only=True
        )

        logger.info(
            f"Loaded collection '{collection}': {len(self._doc_texts)} docs, "
            f"compressed shape {list(self._compressed_docs.shape)}"
        )

    @staticmethod
    def list_collections() -> List[str]:
        """List available CLaRa collections."""
        from fitz_ai.core.paths import FitzPaths

        clara_dir = FitzPaths.workspace() / "clara"
        if not clara_dir.exists():
            return []
        return [
            p.name for p in clara_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists()
        ]

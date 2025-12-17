# fitz/engines/clara/engine.py
"""
ClaraEngine - Knowledge engine implementation for CLaRa paradigm.

This engine wraps Apple's CLaRa (Continuous Latent Reasoning) model behind
the paradigm-agnostic KnowledgeEngine interface.

CLaRa differs from Classic RAG:
- Documents are compressed into continuous memory tokens (16x-128x compression)
- Retrieval happens in latent space via cosine similarity
- Retriever and generator are jointly optimized (end-to-end)
- Single language modeling loss trains both retrieval and generation
"""

from typing import Optional, Dict, Any, List
import logging

from fitz.core import (
    Query,
    Answer,
    Provenance,
    QueryError,
    KnowledgeError,
    GenerationError,
    ConfigurationError,
)
from fitz.engines.clara.config.schema import ClaraConfig

logger = logging.getLogger(__name__)


class ClaraEngine:
    """
    CLaRa (Continuous Latent Reasoning) engine implementation.

    This engine implements the CLaRa paradigm:
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
        >>> from fitz.engines.clara import ClaraEngine
        >>> from fitz.engines.clara.config.schema import ClaraConfig
        >>>
        >>> config = ClaraConfig()
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
        self._compressed_docs: Dict[str, Any] = {}  # doc_id -> compressed embedding
        self._doc_texts: Dict[str, str] = {}  # doc_id -> original text

        try:
            self._initialize_model()
        except Exception as e:
            raise ConfigurationError(f"Failed to initialize CLaRa engine: {e}") from e

    def _initialize_model(self) -> None:
        """
        Load the CLaRa model from HuggingFace.

        CLaRa uses AutoModel with trust_remote_code=True to load
        the custom model architecture.
        """
        try:
            import torch
            from transformers import AutoModel
        except ImportError as e:
            raise ConfigurationError(
                "CLaRa requires transformers and torch. "
                "Install with: pip install transformers torch"
            ) from e

        model_config = self._config.model

        logger.info(f"Loading CLaRa model: {model_config.model_name_or_path}")

        # Determine torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(model_config.torch_dtype, torch.bfloat16)

        # Load model with appropriate settings
        load_kwargs = {
            "trust_remote_code": model_config.trust_remote_code,
            "torch_dtype": torch_dtype,
        }

        if model_config.load_in_8bit:
            load_kwargs["load_in_8bit"] = True
        elif model_config.load_in_4bit:
            load_kwargs["load_in_4bit"] = True

        self._model = AutoModel.from_pretrained(
            model_config.model_name_or_path,
            **load_kwargs
        )

        # Move to device if not using quantization
        if not (model_config.load_in_8bit or model_config.load_in_4bit):
            self._model = self._model.to(model_config.device)

        self._model.eval()
        logger.info(f"CLaRa model loaded on {model_config.device}")

    def add_documents(
            self,
            documents: List[str],
            doc_ids: Optional[List[str]] = None
    ) -> List[str]:
        """
        Add documents to the knowledge base.

        Documents are compressed into continuous memory tokens for efficient
        storage and retrieval.

        Args:
            documents: List of document texts to add
            doc_ids: Optional list of document IDs. Auto-generated if not provided.

        Returns:
            List of document IDs

        Raises:
            KnowledgeError: If compression fails
        """
        if doc_ids is None:
            import uuid
            doc_ids = [f"doc_{uuid.uuid4().hex[:8]}" for _ in documents]

        if len(doc_ids) != len(documents):
            raise ValueError("doc_ids length must match documents length")

        try:
            # Compress documents using CLaRa's compressor
            compressed = self._compress_documents(documents)

            for doc_id, doc_text, embedding in zip(doc_ids, documents, compressed):
                self._compressed_docs[doc_id] = embedding
                self._doc_texts[doc_id] = doc_text

            logger.info(f"Added {len(documents)} documents to CLaRa knowledge base")
            return doc_ids

        except Exception as e:
            raise KnowledgeError(f"Failed to compress documents: {e}") from e

    def _compress_documents(self, documents: List[str]) -> List[Any]:
        """
        Compress documents into continuous memory tokens.

        Uses CLaRa's semantic compressor to convert full documents
        into a small number of continuous embeddings.
        """
        import torch

        # Format documents for compression
        # CLaRa expects documents in a specific format
        formatted_docs = [[doc] for doc in documents]

        with torch.no_grad():
            # Use the model's internal compression
            # The exact method depends on CLaRa variant:
            # - Base: generates paraphrase
            # - Instruct: generates text
            # - E2E: full retrieval + generation

            # For compression, we use the compressor adapter
            if hasattr(self._model, 'compress_documents'):
                compressed = self._model.compress_documents(
                    documents=formatted_docs,
                    max_length=self._config.compression.doc_max_length,
                )
            else:
                # Fallback: use generate_from_paraphrase for compression
                compressed = self._get_document_embeddings(formatted_docs)

        return compressed

    def _get_document_embeddings(self, documents: List[List[str]]) -> List[Any]:
        """
        Get compressed embeddings for documents.

        This uses CLaRa's internal document representation.
        """
        import torch

        # Flatten for processing
        flat_docs = [doc[0] if doc else "" for doc in documents]

        # Get embeddings through model forward pass
        # This captures the compressed representation
        with torch.no_grad():
            # Different variants have different methods
            if hasattr(self._model, 'encode_documents'):
                embeddings = self._model.encode_documents(flat_docs)
            else:
                # Use tokenizer + model hidden states
                tokenizer = self._model.tokenizer if hasattr(self._model, 'tokenizer') else None
                if tokenizer:
                    inputs = tokenizer(
                        flat_docs,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=self._config.compression.doc_max_length
                    ).to(self._model.device)

                    outputs = self._model(**inputs, output_hidden_states=True)
                    # Use last hidden state mean as embedding
                    embeddings = outputs.hidden_states[-1].mean(dim=1)
                else:
                    raise KnowledgeError("Model doesn't have encoding capability")

        return [emb.cpu() for emb in embeddings]

    def answer(self, query: Query) -> Answer:
        """
        Execute a query against knowledge and return an answer.

        CLaRa's answer generation:
        1. Encode query into latent space
        2. Find top-k most similar compressed documents
        3. Generate answer from query + compressed docs

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

        if not self._compressed_docs:
            raise KnowledgeError(
                "No documents in knowledge base. "
                "Call add_documents() first or load a pre-compressed knowledge base."
            )

        try:
            # Step 1: Retrieve relevant documents
            top_k = self._config.retrieval.top_k
            if query.constraints and query.constraints.max_sources:
                top_k = min(top_k, query.constraints.max_sources)

            retrieved_doc_ids, scores = self._retrieve(query.text, top_k=top_k)

            # Step 2: Gather retrieved document texts
            retrieved_docs = [self._doc_texts[doc_id] for doc_id in retrieved_doc_ids]

            # Step 3: Generate answer using CLaRa
            answer_text, used_indices = self._generate(query.text, retrieved_docs)

            # Step 4: Build provenance
            # Note: Provenance takes source_id, excerpt, and metadata (no direct score field)
            provenance = []
            for i, (doc_id, score) in enumerate(zip(retrieved_doc_ids, scores)):
                provenance.append(Provenance(
                    source_id=doc_id,
                    excerpt=self._doc_texts[doc_id][:200] + "..." if len(self._doc_texts[doc_id]) > 200 else
                    self._doc_texts[doc_id],
                    metadata={
                        "rank": i + 1,
                        "relevance_score": float(score),
                        "compression_rate": self._config.compression.compression_rate,
                    }
                ))

            return Answer(
                text=answer_text,
                provenance=provenance,
                metadata={
                    "engine": "clara",
                    "variant": self._config.model.variant,
                    "compression_rate": self._config.compression.compression_rate,
                    "num_docs_retrieved": len(retrieved_doc_ids),
                }
            )

        except QueryError:
            raise
        except KnowledgeError:
            raise
        except Exception as e:
            raise GenerationError(f"CLaRa generation failed: {e}") from e

    def _retrieve(self, query_text: str, top_k: int) -> tuple[List[str], List[float]]:
        """
        Retrieve top-k documents via latent space similarity.

        CLaRa retrieval:
        1. Encode query into same latent space as documents
        2. Compute cosine similarity with all compressed docs
        3. Return top-k most similar
        """
        import torch
        import torch.nn.functional as F

        with torch.no_grad():
            # Encode query
            query_embedding = self._encode_query(query_text)

            # Compute similarities with all documents
            doc_ids = list(self._compressed_docs.keys())
            doc_embeddings = torch.stack([
                self._compressed_docs[doc_id]
                for doc_id in doc_ids
            ])

            # Cosine similarity
            query_embedding = query_embedding.to(doc_embeddings.device)
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(0),
                doc_embeddings,
                dim=-1
            )

            # Get top-k
            top_k = min(top_k, len(doc_ids))
            scores, indices = torch.topk(similarities, top_k)

            retrieved_ids = [doc_ids[i] for i in indices.tolist()]
            retrieved_scores = scores.tolist()

        return retrieved_ids, retrieved_scores

    def _encode_query(self, query_text: str) -> Any:
        """
        Encode query into latent space.

        CLaRa uses a query reasoner adapter to map queries into
        the same continuous space as compressed documents.
        """
        import torch

        with torch.no_grad():
            if hasattr(self._model, 'encode_query'):
                return self._model.encode_query(query_text)
            else:
                # Fallback to tokenize + forward
                tokenizer = getattr(self._model, 'tokenizer', None)
                if tokenizer:
                    inputs = tokenizer(
                        query_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=512
                    ).to(self._model.device)

                    outputs = self._model(**inputs, output_hidden_states=True)
                    return outputs.hidden_states[-1].mean(dim=1).squeeze()
                else:
                    raise KnowledgeError("Cannot encode query without tokenizer")

    def _generate(
            self,
            query_text: str,
            documents: List[str]
    ) -> tuple[str, List[int]]:
        """
        Generate answer from query and retrieved documents.

        Uses CLaRa's unified generation approach where compressed
        document embeddings are concatenated with query tokens.
        """
        import torch

        gen_config = self._config.generation

        with torch.no_grad():
            # Format for CLaRa's generate method
            questions = [query_text]
            docs_batch = [documents]

            # Use appropriate generation method based on variant
            variant = self._config.model.variant

            if variant == "e2e" and hasattr(self._model, 'generate_from_questions'):
                # E2E variant: full retrieval + generation
                outputs, topk_indices = self._model.generate_from_questions(
                    questions=questions,
                    documents=docs_batch,
                    max_new_tokens=gen_config.max_new_tokens,
                )
                return outputs[0], topk_indices.tolist() if hasattr(topk_indices, 'tolist') else topk_indices

            elif variant == "instruct" and hasattr(self._model, 'generate_from_text'):
                # Instruct variant: instruction-following generation
                outputs = self._model.generate_from_text(
                    questions=questions,
                    documents=docs_batch,
                    max_new_tokens=gen_config.max_new_tokens,
                )
                return outputs[0], list(range(len(documents)))

            elif hasattr(self._model, 'generate_from_paraphrase'):
                # Base variant: paraphrase-based generation
                outputs = self._model.generate_from_paraphrase(
                    questions=questions,
                    documents=docs_batch,
                    max_new_tokens=gen_config.max_new_tokens,
                )
                return outputs[0], list(range(len(documents)))

            else:
                raise GenerationError(
                    f"CLaRa variant '{variant}' doesn't have expected generation method"
                )

    def get_knowledge_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge base.

        Returns:
            Dictionary with knowledge base statistics
        """
        return {
            "num_documents": len(self._compressed_docs),
            "compression_rate": self._config.compression.compression_rate,
            "model_variant": self._config.model.variant,
            "cache_enabled": self._config.cache_compressed_docs,
        }

    def clear_knowledge_base(self) -> None:
        """
        Clear all documents from the knowledge base.
        """
        self._compressed_docs.clear()
        self._doc_texts.clear()
        logger.info("CLaRa knowledge base cleared")
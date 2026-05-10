# fitz_sage/evaluation/benchmarks/beir.py
"""
BEIR benchmark integration for evaluating retrieval quality.

BEIR (Benchmarking IR) is an industry-standard benchmark for evaluating
information retrieval systems. It provides standardized datasets and
metrics (nDCG@10, Recall@100) for comparing retrieval quality.

Usage:
    from fitz_sage.evaluation.benchmarks import BEIRBenchmark

    benchmark = BEIRBenchmark(data_dir="~/.fitz/beir_data")
    results = benchmark.evaluate(engine, dataset="scifact")
    print(f"nDCG@10: {results.ndcg_at_10:.4f}")

Requires: pip install fitz-sage[benchmarks]
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fitz_sage.logging.logger import get_logger

if TYPE_CHECKING:
    from fitz_sage.engines.fitz_krag.engine import FitzKragEngine

logger = get_logger(__name__)

# Tier 1 datasets (smallest, fastest for quick testing)
TIER1_DATASETS = ["scifact", "scidocs", "fiqa"]

# Tier 2 datasets (medium size)
TIER2_DATASETS = ["nfcorpus", "arguana", "trec-covid", "webis-touche2020"]

# Tier 3 datasets (large, full benchmark)
TIER3_DATASETS = ["msmarco", "nq", "hotpotqa", "fever", "climate-fever", "dbpedia-entity"]

ALL_DATASETS = TIER1_DATASETS + TIER2_DATASETS + TIER3_DATASETS


@dataclass
class BEIRResult:
    """Results from a single BEIR dataset evaluation."""

    dataset: str
    """Name of the BEIR dataset evaluated."""

    ndcg_at_10: float
    """Normalized Discounted Cumulative Gain at k=10 (primary metric)."""

    ndcg_at_100: float
    """nDCG at k=100."""

    recall_at_10: float
    """Recall at k=10."""

    recall_at_100: float
    """Recall at k=100."""

    map_at_10: float
    """Mean Average Precision at k=10."""

    mrr_at_10: float
    """Mean Reciprocal Rank at k=10."""

    precision_at_10: float
    """Precision at k=10."""

    num_queries: int
    """Number of queries evaluated."""

    num_docs: int
    """Number of documents in corpus."""

    evaluation_time_seconds: float
    """Time taken to run evaluation."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When evaluation was run."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Additional metadata (embedding model, config, etc.)."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "dataset": self.dataset,
            "ndcg_at_10": self.ndcg_at_10,
            "ndcg_at_100": self.ndcg_at_100,
            "recall_at_10": self.recall_at_10,
            "recall_at_100": self.recall_at_100,
            "map_at_10": self.map_at_10,
            "mrr_at_10": self.mrr_at_10,
            "precision_at_10": self.precision_at_10,
            "num_queries": self.num_queries,
            "num_docs": self.num_docs,
            "evaluation_time_seconds": self.evaluation_time_seconds,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BEIRResult:
        """Create from dictionary."""
        data = data.copy()
        data["timestamp"] = datetime.fromisoformat(data["timestamp"])
        return cls(**data)

    def __str__(self) -> str:
        return (
            f"BEIR({self.dataset}): nDCG@10={self.ndcg_at_10:.4f}, "
            f"Recall@100={self.recall_at_100:.4f}"
        )


@dataclass
class BEIRSuiteResult:
    """Results from running multiple BEIR datasets."""

    results: list[BEIRResult]
    """Individual dataset results."""

    average_ndcg_at_10: float
    """Average nDCG@10 across all datasets."""

    average_recall_at_100: float
    """Average Recall@100 across all datasets."""

    total_evaluation_time_seconds: float
    """Total evaluation time."""

    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    """When suite was run."""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "results": [r.to_dict() for r in self.results],
            "average_ndcg_at_10": self.average_ndcg_at_10,
            "average_recall_at_100": self.average_recall_at_100,
            "total_evaluation_time_seconds": self.total_evaluation_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BEIRSuiteResult:
        """Create from dictionary."""
        return cls(
            results=[BEIRResult.from_dict(r) for r in data["results"]],
            average_ndcg_at_10=data["average_ndcg_at_10"],
            average_recall_at_100=data["average_recall_at_100"],
            total_evaluation_time_seconds=data["total_evaluation_time_seconds"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
        )


class FitzBEIRRetriever:
    """
    Adapter that makes Fitz engine compatible with BEIR's retriever interface.

    Indexes the BEIR corpus into a dedicated Fitz collection using vector
    embeddings, then uses vector similarity search to retrieve relevant
    documents for each query.

    BEIR expects a retriever with a `search` method that takes queries and
    returns ranked document IDs with scores.
    """

    def __init__(self, engine: "FitzKragEngine", beir_collection: str, k: int = 100):
        """
        Initialize the adapter.

        Args:
            engine: Fitz RAG engine to use for embedding and search
            beir_collection: Name of the dedicated collection to index the BEIR corpus into
            k: Number of documents to retrieve per query
        """
        self._engine = engine
        self._beir_collection = beir_collection
        self._k = k

    def index_corpus(
        self,
        corpus: dict[str, dict[str, str]],
        batch_size: int = 64,
    ) -> None:
        """
        Index BEIR corpus documents into the dedicated BEIR collection.

        Embeds each document (title + text) and stores it as a section with
        its vector in the BEIR collection. The BEIR doc_id is stored in
        section metadata for retrieval mapping.

        Args:
            corpus: BEIR corpus {doc_id: {"title": ..., "text": ...}}
            batch_size: Number of documents to embed per batch
        """
        from fitz_sage.engines.fitz_krag.ingestion.raw_file_store import RawFileStore
        from fitz_sage.engines.fitz_krag.ingestion.schema import ensure_schema
        from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore

        embedder = self._engine._embedder
        conn_manager = self._engine._connection_manager

        ensure_schema(conn_manager, self._beir_collection, embedder.dimensions)

        raw_store = RawFileStore(conn_manager, self._beir_collection)
        section_store = SectionStore(conn_manager, self._beir_collection)

        doc_ids = list(corpus.keys())
        logger.info(f"Indexing {len(doc_ids)} BEIR documents into '{self._beir_collection}'")

        for i in range(0, len(doc_ids), batch_size):
            batch_ids = doc_ids[i : i + batch_size]
            texts = []
            for doc_id in batch_ids:
                doc = corpus[doc_id]
                title = doc.get("title", "")
                text = doc.get("text", "")
                combined = f"{title}\n\n{text}".strip() if title else text
                # Ensure non-empty text — embed APIs reject empty strings.
                # Truncate to 8000 chars (~2000 tokens) to stay within model context limits.
                safe = (combined if combined.strip() else title or doc_id)[:8000]
                texts.append(safe)

            try:
                vectors = embedder.embed_batch(texts)
            except Exception:
                # Batch failed — fall back to per-document embedding, skipping any that error
                vectors = []
                good_ids: list[str] = []
                good_texts: list[str] = []
                for doc_id, text in zip(batch_ids, texts):
                    try:
                        vectors.append(embedder.embed(text))
                        good_ids.append(doc_id)
                        good_texts.append(text)
                    except Exception as embed_err:
                        logger.warning(f"Skipping doc {doc_id}: embed failed ({embed_err})")
                batch_ids = good_ids
                texts = good_texts
                if not vectors:
                    continue

            # Insert raw files first — SectionStore has FK on raw_file_id
            for doc_id, text in zip(batch_ids, texts):
                raw_store.upsert(
                    file_id=doc_id,
                    path=f"beir/{doc_id}",
                    content=text,
                    content_hash=hashlib.sha256(text.encode()).hexdigest(),
                    file_type="beir_doc",
                    size_bytes=len(text.encode()),
                    metadata={"beir_doc_id": doc_id},
                )

            sections = []
            for j, (doc_id, text, vector) in enumerate(zip(batch_ids, texts, vectors)):
                sections.append(
                    {
                        "id": f"sec_{doc_id}",
                        "raw_file_id": doc_id,
                        "title": corpus[doc_id].get("title", ""),
                        "level": 1,
                        "page_start": None,
                        "page_end": None,
                        "content": text,
                        "summary": text[:2000],
                        "summary_vector": vector,
                        "parent_section_id": None,
                        "position": i + j,
                        "keywords": [],
                        "entities": [],
                        "metadata": {"doc_id": doc_id},
                    }
                )
            section_store.upsert_batch(sections)

        logger.info(f"Indexed {len(doc_ids)} BEIR documents")

    def search(
        self,
        corpus: dict[str, dict[str, str]],
        queries: dict[str, str],
        top_k: int,
        **kwargs,
    ) -> dict[str, dict[str, float]]:
        """
        Search for relevant documents using BM25 + semantic hybrid retrieval.

        Uses SectionSearchStrategy which combines PostgreSQL BM25 full-text
        search (0.6 weight) with dense vector cosine similarity (0.4 weight),
        matching Fitz's production retrieval behaviour.

        Args:
            corpus: Document corpus (unused; already indexed via index_corpus)
            queries: Query corpus {query_id: query_text}
            top_k: Number of results to return per query

        Returns:
            Results as {query_id: {doc_id: score}}
        """
        from fitz_sage.engines.fitz_krag.ingestion.section_store import SectionStore
        from fitz_sage.engines.fitz_krag.retrieval.strategies.section_search import (
            SectionSearchStrategy,
        )

        section_store = SectionStore(self._engine._connection_manager, self._beir_collection)
        strategy = SectionSearchStrategy(section_store, self._engine._config)

        results: dict[str, dict[str, float]] = {}

        for query_id, query_text in queries.items():
            if not query_text or not query_text.strip():
                results[query_id] = {}
                continue
            try:
                addresses = strategy.retrieve(query_text, top_k)

                # Address.source_id == raw_file_id == BEIR doc_id (set in index_corpus)
                query_results: dict[str, float] = {}
                for addr in addresses:
                    doc_id = addr.source_id
                    if doc_id and doc_id not in query_results:
                        query_results[doc_id] = addr.score

                results[query_id] = query_results

            except Exception as e:
                logger.warning(f"Error retrieving for query {query_id}: {e}")
                results[query_id] = {}

        return results

    def cleanup(self) -> None:
        """Remove all indexed documents from the BEIR collection."""
        try:
            conn_manager = self._engine._connection_manager
            with conn_manager.connection(self._beir_collection) as conn:
                conn.execute("DELETE FROM krag_section_index")
                conn.execute("DELETE FROM krag_raw_files")
                conn.commit()
            logger.info(f"Cleaned up BEIR collection '{self._beir_collection}'")
        except Exception as e:
            logger.warning(f"BEIR cleanup failed for '{self._beir_collection}': {e}")


class BEIRBenchmark:
    """
    BEIR benchmark runner for Fitz.

    Downloads and caches BEIR datasets, indexes them into a dedicated
    collection, and evaluates retrieval quality using nDCG, Recall, MAP,
    MRR, and Precision metrics.

    Example:
        benchmark = BEIRBenchmark()

        # Evaluate on a single dataset
        result = benchmark.evaluate(engine, "scifact")

        # Evaluate on multiple datasets
        suite_result = benchmark.evaluate_suite(engine, ["scifact", "fiqa"])
    """

    def __init__(self, data_dir: str | Path | None = None):
        """
        Initialize BEIR benchmark.

        Args:
            data_dir: Directory to store downloaded datasets.
                     Defaults to ~/.fitz/beir_data/
        """
        if data_dir is None:
            data_dir = Path.home() / ".fitz" / "beir_data"
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def evaluate(
        self,
        engine: "FitzKragEngine",
        dataset: str,
        split: str = "test",
        k_values: list[int] | None = None,
        **kwargs,
    ) -> BEIRResult:
        """
        Evaluate retrieval quality on a BEIR dataset.

        Downloads the dataset if not cached, indexes the corpus into a
        dedicated collection, runs all queries, then cleans up.

        Args:
            engine: Fitz RAG engine to evaluate
            dataset: BEIR dataset name (e.g., "scifact", "nfcorpus")
            split: Dataset split to use ("test", "dev", "train")
            k_values: Values of k for metrics (default: [10, 100])

        Returns:
            BEIRResult with metrics

        Raises:
            ImportError: If beir package is not installed
        """
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader
            from beir.retrieval.evaluation import EvaluateRetrieval
        except ImportError as e:
            raise ImportError(
                "BEIR benchmark requires the 'beir' package. "
                "Install with: pip install fitz-sage[benchmarks]"
            ) from e

        import time
        import uuid

        if k_values is None:
            k_values = [10, 100]

        start_time = time.time()

        # Download dataset if not cached
        dataset_path = self._data_dir / dataset
        if not dataset_path.exists():
            logger.info(f"Downloading BEIR dataset: {dataset}")
            url = (
                f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset}.zip"
            )
            util.download_and_unzip(url, str(self._data_dir))

        # Load dataset
        logger.info(f"Loading BEIR dataset: {dataset}")
        corpus, queries, qrels = GenericDataLoader(str(dataset_path)).load(split=split)

        # Use a unique collection name to avoid conflicts across concurrent evaluations
        beir_collection = f"beir_{dataset}_{uuid.uuid4().hex[:8]}"
        retriever = FitzBEIRRetriever(engine, beir_collection, k=max(k_values))

        try:
            # Index corpus before running queries
            logger.info(f"Indexing {len(corpus)} documents for BEIR evaluation")
            retriever.index_corpus(corpus)

            # Run retrieval
            logger.info(f"Running retrieval on {len(queries)} queries")
            results = retriever.search(corpus, queries, max(k_values))
        finally:
            retriever.cleanup()

        # Evaluate
        evaluator = EvaluateRetrieval()
        ndcg, _map, recall, precision = evaluator.evaluate(
            qrels, results, k_values, ignore_identical_ids=True
        )

        # Calculate MRR
        mrr = evaluator.evaluate_custom(qrels, results, k_values, metric="mrr")

        evaluation_time = time.time() - start_time

        return BEIRResult(
            dataset=dataset,
            ndcg_at_10=ndcg.get("NDCG@10", 0.0),
            ndcg_at_100=ndcg.get("NDCG@100", 0.0),
            recall_at_10=recall.get("Recall@10", 0.0),
            recall_at_100=recall.get("Recall@100", 0.0),
            map_at_10=_map.get("MAP@10", 0.0),
            mrr_at_10=mrr.get("MRR@10", 0.0),
            precision_at_10=precision.get("P@10", 0.0),
            num_queries=len(queries),
            num_docs=len(corpus),
            evaluation_time_seconds=evaluation_time,
            metadata={
                "split": split,
                "k_values": k_values,
            },
        )

    def evaluate_suite(
        self,
        engine: "FitzKragEngine",
        datasets: list[str] | None = None,
        split: str = "test",
    ) -> BEIRSuiteResult:
        """
        Evaluate on multiple BEIR datasets.

        Args:
            engine: Fitz RAG engine to evaluate
            datasets: List of dataset names. Defaults to TIER1_DATASETS.
            split: Dataset split to use

        Returns:
            BEIRSuiteResult with aggregated metrics
        """
        if datasets is None:
            datasets = TIER1_DATASETS

        import time

        start_time = time.time()
        results: list[BEIRResult] = []

        for dataset in datasets:
            logger.info(f"Evaluating BEIR dataset: {dataset}")
            try:
                result = self.evaluate(engine, dataset, split)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to evaluate {dataset}: {e}")

        total_time = time.time() - start_time

        if results:
            avg_ndcg = sum(r.ndcg_at_10 for r in results) / len(results)
            avg_recall = sum(r.recall_at_100 for r in results) / len(results)
        else:
            avg_ndcg = 0.0
            avg_recall = 0.0

        return BEIRSuiteResult(
            results=results,
            average_ndcg_at_10=avg_ndcg,
            average_recall_at_100=avg_recall,
            total_evaluation_time_seconds=total_time,
        )

    def save_results(self, result: BEIRResult | BEIRSuiteResult, path: Path | str) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Saved BEIR results to {path}")

    def load_results(self, path: Path | str) -> BEIRResult | BEIRSuiteResult:
        """Load results from JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        if "results" in data:
            return BEIRSuiteResult.from_dict(data)
        return BEIRResult.from_dict(data)

    @staticmethod
    def get_available_datasets() -> dict[str, list[str]]:
        """Get available BEIR datasets by tier."""
        return {
            "tier1": TIER1_DATASETS,
            "tier2": TIER2_DATASETS,
            "tier3": TIER3_DATASETS,
        }

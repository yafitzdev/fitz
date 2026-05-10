# tests/unit/test_beir_benchmark.py
"""
Unit tests for the BEIR benchmark module.

Tests:
- BEIRResult: serialization, from_dict round-trip
- BEIRSuiteResult: serialization, from_dict round-trip
- BEIRBenchmark: dataset listing, save/load results
- FitzBEIRRetriever: search with mocked section_store, cleanup error handling
- BEIRBenchmark.evaluate: ImportError when beir package missing
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from fitz_sage.evaluation.benchmarks.beir import (
    ALL_DATASETS,
    TIER1_DATASETS,
    TIER2_DATASETS,
    TIER3_DATASETS,
    BEIRBenchmark,
    BEIRResult,
    BEIRSuiteResult,
    FitzBEIRRetriever,
)

# =============================================================================
# BEIRResult Tests
# =============================================================================


class TestBEIRResult:
    """Tests for BEIRResult dataclass."""

    def _make_result(self, dataset: str = "scifact", **overrides) -> BEIRResult:
        defaults = dict(
            dataset=dataset,
            ndcg_at_10=0.6832,
            ndcg_at_100=0.7124,
            recall_at_10=0.5891,
            recall_at_100=0.8234,
            map_at_10=0.6012,
            mrr_at_10=0.7345,
            precision_at_10=0.0823,
            num_queries=300,
            num_docs=5183,
            evaluation_time_seconds=42.7,
        )
        defaults.update(overrides)
        return BEIRResult(**defaults)

    def test_to_dict_contains_all_fields(self):
        result = self._make_result()
        d = result.to_dict()

        assert d["dataset"] == "scifact"
        assert d["ndcg_at_10"] == pytest.approx(0.6832)
        assert d["ndcg_at_100"] == pytest.approx(0.7124)
        assert d["recall_at_10"] == pytest.approx(0.5891)
        assert d["recall_at_100"] == pytest.approx(0.8234)
        assert d["map_at_10"] == pytest.approx(0.6012)
        assert d["mrr_at_10"] == pytest.approx(0.7345)
        assert d["precision_at_10"] == pytest.approx(0.0823)
        assert d["num_queries"] == 300
        assert d["num_docs"] == 5183
        assert d["evaluation_time_seconds"] == pytest.approx(42.7)
        assert "timestamp" in d
        assert "metadata" in d

    def test_from_dict_round_trip(self):
        original = self._make_result()
        d = original.to_dict()
        restored = BEIRResult.from_dict(d)

        assert restored.dataset == original.dataset
        assert restored.ndcg_at_10 == pytest.approx(original.ndcg_at_10)
        assert restored.recall_at_100 == pytest.approx(original.recall_at_100)
        assert restored.num_queries == original.num_queries
        assert restored.num_docs == original.num_docs

    def test_from_dict_parses_timestamp(self):
        result = self._make_result()
        d = result.to_dict()
        restored = BEIRResult.from_dict(d)
        assert isinstance(restored.timestamp, datetime)

    def test_str_contains_dataset_and_metrics(self):
        result = self._make_result(dataset="nfcorpus")
        s = str(result)
        assert "nfcorpus" in s
        assert "nDCG@10" in s
        assert "Recall@100" in s

    def test_metadata_preserved_in_round_trip(self):
        result = self._make_result(metadata={"split": "test", "k_values": [10, 100]})
        d = result.to_dict()
        restored = BEIRResult.from_dict(d)
        assert restored.metadata["split"] == "test"
        assert restored.metadata["k_values"] == [10, 100]


# =============================================================================
# BEIRSuiteResult Tests
# =============================================================================


class TestBEIRSuiteResult:
    """Tests for BEIRSuiteResult dataclass."""

    def _make_suite(self) -> BEIRSuiteResult:
        results = [
            BEIRResult(
                dataset="scifact",
                ndcg_at_10=0.70,
                ndcg_at_100=0.74,
                recall_at_10=0.60,
                recall_at_100=0.82,
                map_at_10=0.65,
                mrr_at_10=0.73,
                precision_at_10=0.08,
                num_queries=300,
                num_docs=5183,
                evaluation_time_seconds=40.0,
            ),
            BEIRResult(
                dataset="fiqa",
                ndcg_at_10=0.50,
                ndcg_at_100=0.54,
                recall_at_10=0.40,
                recall_at_100=0.62,
                map_at_10=0.45,
                mrr_at_10=0.53,
                precision_at_10=0.06,
                num_queries=648,
                num_docs=57638,
                evaluation_time_seconds=120.0,
            ),
        ]
        return BEIRSuiteResult(
            results=results,
            average_ndcg_at_10=0.60,
            average_recall_at_100=0.72,
            total_evaluation_time_seconds=160.0,
        )

    def test_to_dict_contains_all_fields(self):
        suite = self._make_suite()
        d = suite.to_dict()

        assert len(d["results"]) == 2
        assert d["average_ndcg_at_10"] == pytest.approx(0.60)
        assert d["average_recall_at_100"] == pytest.approx(0.72)
        assert d["total_evaluation_time_seconds"] == pytest.approx(160.0)
        assert "timestamp" in d

    def test_from_dict_round_trip(self):
        original = self._make_suite()
        d = original.to_dict()
        restored = BEIRSuiteResult.from_dict(d)

        assert len(restored.results) == 2
        assert restored.results[0].dataset == "scifact"
        assert restored.results[1].dataset == "fiqa"
        assert restored.average_ndcg_at_10 == pytest.approx(original.average_ndcg_at_10)
        assert restored.average_recall_at_100 == pytest.approx(original.average_recall_at_100)
        assert restored.total_evaluation_time_seconds == pytest.approx(
            original.total_evaluation_time_seconds
        )

    def test_from_dict_parses_timestamp(self):
        suite = self._make_suite()
        d = suite.to_dict()
        restored = BEIRSuiteResult.from_dict(d)
        assert isinstance(restored.timestamp, datetime)


# =============================================================================
# BEIRBenchmark Tests
# =============================================================================


class TestBEIRBenchmark:
    """Tests for BEIRBenchmark."""

    def test_get_available_datasets_returns_all_tiers(self):
        datasets = BEIRBenchmark.get_available_datasets()
        assert "tier1" in datasets
        assert "tier2" in datasets
        assert "tier3" in datasets
        assert datasets["tier1"] == TIER1_DATASETS
        assert datasets["tier2"] == TIER2_DATASETS
        assert datasets["tier3"] == TIER3_DATASETS

    def test_default_data_dir(self):
        benchmark = BEIRBenchmark()
        assert ".fitz" in str(benchmark._data_dir)
        assert "beir_data" in str(benchmark._data_dir)

    def test_custom_data_dir(self, tmp_path):
        benchmark = BEIRBenchmark(data_dir=tmp_path / "custom_beir")
        assert benchmark._data_dir == tmp_path / "custom_beir"
        assert benchmark._data_dir.exists()

    def test_save_and_load_single_result(self, tmp_path):
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        result = BEIRResult(
            dataset="scifact",
            ndcg_at_10=0.68,
            ndcg_at_100=0.71,
            recall_at_10=0.58,
            recall_at_100=0.82,
            map_at_10=0.60,
            mrr_at_10=0.73,
            precision_at_10=0.08,
            num_queries=300,
            num_docs=5183,
            evaluation_time_seconds=42.0,
        )

        output_path = tmp_path / "result.json"
        benchmark.save_results(result, output_path)

        assert output_path.exists()
        loaded = benchmark.load_results(output_path)
        assert isinstance(loaded, BEIRResult)
        assert loaded.dataset == "scifact"
        assert loaded.ndcg_at_10 == pytest.approx(0.68)

    def test_save_and_load_suite_result(self, tmp_path):
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        suite = BEIRSuiteResult(
            results=[
                BEIRResult(
                    dataset="scifact",
                    ndcg_at_10=0.68,
                    ndcg_at_100=0.71,
                    recall_at_10=0.58,
                    recall_at_100=0.82,
                    map_at_10=0.60,
                    mrr_at_10=0.73,
                    precision_at_10=0.08,
                    num_queries=300,
                    num_docs=5183,
                    evaluation_time_seconds=42.0,
                )
            ],
            average_ndcg_at_10=0.68,
            average_recall_at_100=0.82,
            total_evaluation_time_seconds=42.0,
        )

        output_path = tmp_path / "suite.json"
        benchmark.save_results(suite, output_path)

        loaded = benchmark.load_results(output_path)
        assert isinstance(loaded, BEIRSuiteResult)
        assert len(loaded.results) == 1
        assert loaded.results[0].dataset == "scifact"

    def test_save_creates_parent_directories(self, tmp_path):
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        result = BEIRResult(
            dataset="scifact",
            ndcg_at_10=0.5,
            ndcg_at_100=0.6,
            recall_at_10=0.4,
            recall_at_100=0.7,
            map_at_10=0.45,
            mrr_at_10=0.55,
            precision_at_10=0.05,
            num_queries=100,
            num_docs=1000,
            evaluation_time_seconds=10.0,
        )

        deep_path = tmp_path / "nested" / "dir" / "result.json"
        benchmark.save_results(result, deep_path)
        assert deep_path.exists()

    def test_evaluate_raises_import_error_without_beir(self, tmp_path):
        """evaluate() raises ImportError when 'beir' package is not installed."""
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        mock_engine = MagicMock()

        with patch.dict("sys.modules", {"beir": None, "beir.util": None}):
            with pytest.raises(ImportError, match="beir"):
                benchmark.evaluate(mock_engine, "scifact")

    def test_evaluate_suite_empty_datasets(self, tmp_path):
        """evaluate_suite with all failures returns zero averages."""
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        mock_engine = MagicMock()

        # Patch evaluate to always fail
        with patch.object(benchmark, "evaluate", side_effect=Exception("network error")):
            suite = benchmark.evaluate_suite(mock_engine, datasets=["scifact"])

        assert suite.results == []
        assert suite.average_ndcg_at_10 == 0.0
        assert suite.average_recall_at_100 == 0.0

    def test_evaluate_suite_averages_correctly(self, tmp_path):
        """evaluate_suite averages nDCG and recall across datasets."""
        benchmark = BEIRBenchmark(data_dir=tmp_path)
        mock_engine = MagicMock()

        result_a = BEIRResult(
            dataset="a",
            ndcg_at_10=0.60,
            ndcg_at_100=0.64,
            recall_at_10=0.50,
            recall_at_100=0.70,
            map_at_10=0.55,
            mrr_at_10=0.63,
            precision_at_10=0.07,
            num_queries=100,
            num_docs=1000,
            evaluation_time_seconds=10.0,
        )
        result_b = BEIRResult(
            dataset="b",
            ndcg_at_10=0.40,
            ndcg_at_100=0.44,
            recall_at_10=0.30,
            recall_at_100=0.50,
            map_at_10=0.35,
            mrr_at_10=0.43,
            precision_at_10=0.05,
            num_queries=200,
            num_docs=2000,
            evaluation_time_seconds=20.0,
        )

        with patch.object(benchmark, "evaluate", side_effect=[result_a, result_b]):
            suite = benchmark.evaluate_suite(mock_engine, datasets=["a", "b"])

        assert suite.average_ndcg_at_10 == pytest.approx(0.50)
        assert suite.average_recall_at_100 == pytest.approx(0.60)
        assert len(suite.results) == 2


# =============================================================================
# FitzBEIRRetriever Tests
# =============================================================================


class TestFitzBEIRRetriever:
    """Tests for FitzBEIRRetriever."""

    def _make_mock_engine(self):
        return MagicMock()

    def _make_address(self, source_id: str, score: float):
        from fitz_sage.engines.fitz_krag.types import Address, AddressKind

        return Address(
            kind=AddressKind.SECTION,
            source_id=source_id,
            location=source_id,
            summary="",
            score=score,
        )

    def test_search_maps_doc_ids_from_metadata(self):
        """search() maps Address.source_id to BEIR doc_id format."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        mock_addresses = [
            self._make_address("doc1", 0.95),
            self._make_address("doc2", 0.87),
            self._make_address("doc3", 0.72),
        ]

        with patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy.retrieve",
            return_value=mock_addresses,
        ):
            results = retriever.search(
                corpus={},
                queries={"q1": "what is the treatment for X?"},
                top_k=10,
            )

        assert "q1" in results
        assert results["q1"]["doc1"] == pytest.approx(0.95)
        assert results["q1"]["doc2"] == pytest.approx(0.87)
        assert results["q1"]["doc3"] == pytest.approx(0.72)

    def test_search_deduplicates_same_doc_id(self):
        """search() keeps only the first (highest-score) hit per source_id."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        mock_addresses = [
            self._make_address("doc1", 0.95),
            self._make_address("doc1", 0.50),  # duplicate
        ]

        with patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy.retrieve",
            return_value=mock_addresses,
        ):
            results = retriever.search(corpus={}, queries={"q1": "query"}, top_k=10)

        assert results["q1"]["doc1"] == pytest.approx(0.95)
        assert len(results["q1"]) == 1

    def test_search_handles_missing_metadata(self):
        """search() skips addresses with empty source_id."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        mock_addresses = [
            self._make_address("", 0.95),  # empty source_id → skipped
            self._make_address("doc2", 0.87),
        ]

        with patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy.retrieve",
            return_value=mock_addresses,
        ):
            results = retriever.search(corpus={}, queries={"q1": "query"}, top_k=10)

        assert "doc2" in results["q1"]
        assert len(results["q1"]) == 1

    def test_search_returns_empty_on_retrieval_error(self):
        """search() returns empty dict for a query if retrieval raises."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        with patch(
            "fitz_sage.engines.fitz_krag.retrieval.strategies.section_search.SectionSearchStrategy.retrieve",
            side_effect=RuntimeError("retrieval failed"),
        ):
            results = retriever.search(corpus={}, queries={"q1": "query"}, top_k=10)

        assert results["q1"] == {}

    def test_search_handles_multiple_queries(self):
        """search() processes all queries independently."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        def search_side_effect(vector, limit):
            # Return different results for different call numbers
            call_count = search_side_effect.count
            search_side_effect.count += 1
            if call_count == 0:
                return [{"metadata": {"doc_id": "doc1"}, "score": 0.9}]
            return [{"metadata": {"doc_id": "doc2"}, "score": 0.8}]

        search_side_effect.count = 0

        with patch(
            "fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore"
        ) as MockSectionStore:
            MockSectionStore.return_value.search_by_vector.side_effect = search_side_effect

            results = retriever.search(
                corpus={},
                queries={"q1": "first query", "q2": "second query"},
                top_k=5,
            )

        assert "q1" in results
        assert "q2" in results

    def test_cleanup_logs_warning_on_error(self, caplog):
        """cleanup() logs a warning if deletion fails instead of raising."""
        engine = self._make_mock_engine()
        engine._connection_manager.connection.side_effect = Exception("db error")
        retriever = FitzBEIRRetriever(engine, "beir_test_col")

        import logging

        with caplog.at_level(logging.WARNING):
            retriever.cleanup()  # Should not raise

        assert any("cleanup failed" in r.message.lower() for r in caplog.records)

    def test_search_uses_correct_collection(self):
        """search() creates SectionStore with the beir_collection name."""
        engine = self._make_mock_engine()
        retriever = FitzBEIRRetriever(engine, "beir_mycollection_abc123")

        with patch(
            "fitz_sage.engines.fitz_krag.ingestion.section_store.SectionStore"
        ) as MockSectionStore:
            MockSectionStore.return_value.search_by_vector.return_value = []
            retriever.search(corpus={}, queries={"q1": "query"}, top_k=5)

        MockSectionStore.assert_called_once_with(
            engine._connection_manager, "beir_mycollection_abc123"
        )


# =============================================================================
# Dataset Constant Tests
# =============================================================================


class TestDatasetConstants:
    def test_all_datasets_is_union_of_tiers(self):
        assert set(ALL_DATASETS) == set(TIER1_DATASETS + TIER2_DATASETS + TIER3_DATASETS)

    def test_tier1_has_expected_datasets(self):
        assert "scifact" in TIER1_DATASETS
        assert "fiqa" in TIER1_DATASETS

    def test_tier3_has_large_datasets(self):
        assert "msmarco" in TIER3_DATASETS
        assert "nq" in TIER3_DATASETS

# tests/unit/test_detection_classifier.py
"""
Unit tests for DetectionClassifier and its integration with DetectionOrchestrator.

Covers:
- Model loading (available / unavailable states)
- ML predictions for temporal and comparison
- Keyword rules for aggregation, freshness, rewriter
- Fail-open behaviour on prediction errors
- DetectionOrchestrator gating: LLM skipped, subset called, all called
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

try:
    import scipy  # noqa: F401

    _has_scipy = True
except ImportError:
    _has_scipy = False

from fitz_sage.retrieval.detection.classifier import DetectionClassifier
from fitz_sage.retrieval.detection.protocol import DetectionCategory
from fitz_sage.retrieval.detection.registry import DetectionOrchestrator

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_classifier_unavailable() -> DetectionClassifier:
    """Return a DetectionClassifier that failed to load its model."""
    clf = DetectionClassifier.__new__(DetectionClassifier)
    clf._model = None
    clf._vectorizer = None
    clf._labels = []
    clf._thresholds = {}
    clf._available = False
    return clf


def _make_classifier_available(
    labels: list[str] | None = None,
    thresholds: dict[str, float] | None = None,
    predict_positive: list[str] | None = None,
) -> DetectionClassifier:
    """
    Return a DetectionClassifier backed by mock model/vectorizer.

    Args:
        labels: Label names (default: ["temporal", "comparison"]).
        thresholds: Per-label thresholds (default: 0.5 for all).
        predict_positive: Labels whose mock model outputs p=0.9 (rest p=0.1).
    """
    labels = labels or ["temporal", "comparison"]
    thresholds = thresholds or {lbl: 0.5 for lbl in labels}
    predict_positive = predict_positive or []

    # Build mock estimators: one per label
    estimators = []
    for label in labels:
        est = MagicMock()
        p_pos = 0.9 if label in predict_positive else 0.1
        # predict_proba returns shape (n_samples, 2): [[p_neg, p_pos]]
        est.predict_proba.return_value = [[1 - p_pos, p_pos]]
        estimators.append(est)

    mock_model = MagicMock()
    mock_model.estimators_ = estimators

    # predict_proba on MultiOutputClassifier returns list of (n,2) arrays

    mock_model.predict_proba.return_value = [
        [[1 - (0.9 if lbl in predict_positive else 0.1), (0.9 if lbl in predict_positive else 0.1)]]
        for lbl in labels
    ]

    mock_vec = MagicMock()
    mock_vec.transform.return_value = MagicMock()  # sparse matrix placeholder

    clf = DetectionClassifier.__new__(DetectionClassifier)
    clf._model = mock_model
    clf._vectorizer = mock_vec
    clf._labels = labels
    clf._thresholds = thresholds
    clf._available = True
    return clf


# ---------------------------------------------------------------------------
# DetectionClassifier — model loading
# ---------------------------------------------------------------------------


class TestDetectionClassifierLoading:
    def test_unavailable_when_model_missing(self, tmp_path):
        """Classifier marks itself unavailable when joblib file is not found."""
        with patch(
            "fitz_sage.retrieval.detection.classifier._MODEL_PATH",
            tmp_path / "nonexistent.joblib",
        ):
            clf = DetectionClassifier()
        assert not clf.available

    def test_predict_returns_none_when_unavailable(self):
        clf = _make_classifier_unavailable()
        result = clf.predict("When did the project launch?")
        assert result is None

    def test_available_true_when_loaded(self):
        clf = _make_classifier_available()
        assert clf.available


# ---------------------------------------------------------------------------
# DetectionClassifier — ML predictions (temporal, comparison)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed; ML predictions unavailable")
class TestDetectionClassifierML:
    def test_temporal_query_flagged(self):
        """Temporal query above threshold returns TEMPORAL in flagged set."""
        clf = _make_classifier_available(predict_positive=["temporal"])
        result = clf.predict("some query")
        assert result is not None
        assert DetectionCategory.TEMPORAL in result

    def test_comparison_query_flagged(self):
        """Comparison query above threshold returns COMPARISON in flagged set."""
        clf = _make_classifier_available(predict_positive=["comparison"])
        result = clf.predict("some query")
        assert result is not None
        assert DetectionCategory.COMPARISON in result

    def test_factual_query_not_ml_flagged(self):
        """Query below threshold for all ML labels not in flagged set (ML part)."""
        clf = _make_classifier_available(predict_positive=[])
        # Use a query with no keyword triggers either
        result = clf.predict("Explain the authentication flow")
        assert result is not None
        assert DetectionCategory.TEMPORAL not in result
        assert DetectionCategory.COMPARISON not in result

    def test_both_labels_flagged(self):
        """Both temporal and comparison flagged simultaneously."""
        clf = _make_classifier_available(predict_positive=["temporal", "comparison"])
        result = clf.predict("some query")
        assert result is not None
        assert DetectionCategory.TEMPORAL in result
        assert DetectionCategory.COMPARISON in result

    def test_threshold_gates_prediction(self):
        """Label with p=0.9 but threshold=0.95 is NOT flagged."""
        clf = _make_classifier_available(
            predict_positive=["temporal"],
            thresholds={"temporal": 0.95, "comparison": 0.5},
        )
        result = clf.predict("some query without temporal keywords")
        assert result is not None
        assert DetectionCategory.TEMPORAL not in result


# ---------------------------------------------------------------------------
# DetectionClassifier — keyword rules
# ---------------------------------------------------------------------------


class TestDetectionClassifierKeywords:
    def _factual_clf(self) -> DetectionClassifier:
        """Classifier that never ML-flags anything."""
        return _make_classifier_available(predict_positive=[])

    @pytest.mark.parametrize(
        "query",
        [
            "List all the failed test cases",
            "How many errors occurred?",
            "Show all open issues",
            "Enumerate the configuration options",
            "What are all the supported formats?",
        ],
    )
    def test_aggregation_keyword_detected(self, query):
        clf = self._factual_clf()
        result = clf.predict(query)
        assert result is not None
        assert DetectionCategory.AGGREGATION in result

    @pytest.mark.parametrize(
        "query",
        [
            "What is the latest version?",
            "Show me the most recent results",
            "What is current best practice?",
            "Give me today's status",
            "What's the newest approach?",
        ],
    )
    def test_freshness_keyword_detected(self, query):
        clf = self._factual_clf()
        result = clf.predict(query)
        assert result is not None
        assert DetectionCategory.FRESHNESS in result

    @pytest.mark.parametrize(
        "query",
        [
            "What does it do?",
            "Tell me more about that",
            "How does this work?",
            "What did they decide?",
            "Explain the previous point",
        ],
    )
    def test_rewriter_keyword_detected(self, query):
        clf = self._factual_clf()
        result = clf.predict(query)
        assert result is not None
        assert DetectionCategory.REWRITER in result

    def test_clean_factual_returns_empty_set(self):
        """A plain factual query with no keyword triggers returns empty set."""
        clf = self._factual_clf()
        result = clf.predict("Explain the OAuth2 authorization code flow")
        assert result is not None
        assert len(result) == 0  # empty set → skip LLM

    def test_multiple_keyword_categories_combined(self):
        """Query triggering both freshness and aggregation gets both."""
        clf = self._factual_clf()
        result = clf.predict("List all the latest configuration options")
        assert result is not None
        assert DetectionCategory.AGGREGATION in result
        assert DetectionCategory.FRESHNESS in result


# ---------------------------------------------------------------------------
# DetectionClassifier — fail-open
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed; ML error paths untestable")
class TestDetectionClassifierFailOpen:
    def test_returns_none_on_vectorizer_error(self):
        """Prediction error returns None (fail-open → caller runs all modules)."""
        clf = _make_classifier_available(predict_positive=["temporal"])
        clf._vectorizer.transform.side_effect = RuntimeError("vectorizer exploded")
        result = clf.predict("When did the project start?")
        assert result is None

    def test_returns_none_on_model_error(self):
        """Model predict_proba error returns None (fail-open)."""
        clf = _make_classifier_available(predict_positive=["temporal"])
        clf._model.predict_proba.side_effect = ValueError("model exploded")
        result = clf.predict("Compare React and Vue")
        assert result is None


# ---------------------------------------------------------------------------
# DetectionOrchestrator — LLM gating
# ---------------------------------------------------------------------------


class TestDetectionOrchestratorGating:
    def _make_orchestrator(self, ml_result: set | None) -> tuple[DetectionOrchestrator, MagicMock]:
        """
        Build a DetectionOrchestrator with a mocked ML classifier and LLM classifier.

        Returns orchestrator and the mock LLM classifier so tests can inspect calls.
        """
        mock_clf = MagicMock()
        mock_clf.available = True
        mock_clf.predict.return_value = ml_result

        mock_llm = MagicMock()
        mock_llm.classify.return_value = {}

        orch = DetectionOrchestrator.__new__(DetectionOrchestrator)
        orch.chat_factory = MagicMock()
        orch._classifier = mock_llm
        orch._ml_classifier = mock_clf

        return orch, mock_llm

    def test_llm_skipped_when_empty_set(self):
        """Empty set from ML classifier → LLM classify is NOT called."""
        orch, mock_llm = self._make_orchestrator(ml_result=set())
        orch.detect_for_retrieval("What is the authentication flow?")
        mock_llm.classify.assert_not_called()

    def test_llm_called_with_subset_when_flagged(self):
        """Non-empty set from ML classifier → LLM called with limit_to."""
        orch, mock_llm = self._make_orchestrator(ml_result={DetectionCategory.TEMPORAL})
        orch.detect_for_retrieval("When did the project launch?")
        mock_llm.classify.assert_called_once()
        _, kwargs = mock_llm.classify.call_args
        assert kwargs.get("limit_to") == {DetectionCategory.TEMPORAL}

    def test_llm_called_with_all_when_classifier_unavailable(self):
        """None from ML classifier → LLM called with limit_to=None (all modules)."""
        orch, mock_llm = self._make_orchestrator(ml_result=None)
        orch.detect_for_retrieval("What is the authentication flow?")
        mock_llm.classify.assert_called_once()
        _, kwargs = mock_llm.classify.call_args
        assert kwargs.get("limit_to") is None

    def test_summary_has_no_temporal_when_llm_skipped(self):
        """When LLM is skipped, temporal detection is not-detected."""
        orch, _ = self._make_orchestrator(ml_result=set())
        summary = orch.detect_for_retrieval("Explain OAuth2")
        assert not summary.has_temporal_intent

    def test_summary_has_no_comparison_when_llm_skipped(self):
        """When LLM is skipped, comparison detection is not-detected."""
        orch, _ = self._make_orchestrator(ml_result=set())
        summary = orch.detect_for_retrieval("Explain OAuth2")
        assert not summary.has_comparison_intent

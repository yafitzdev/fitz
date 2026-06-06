# tests/unit/test_pyrrho.py
"""Unit tests for the Pyrrho governance backend."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from fitz_sage.core.answer_mode import AnswerMode
from fitz_sage.governance.pyrrho import (
    GovernanceDecision,
    Pyrrho,
    QueryDecision,
    _decision_from_outputs,
    _head_decision,
    _load_trustworthy_threshold,
)


def test_decide_many_batches_non_empty_prefixes():
    """decide_many should batch non-empty prefixes and preserve empty-prefix abstain."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._predict_context_batches = MagicMock(
        return_value=[
            GovernanceDecision(
                mode=AnswerMode.ABSTAIN,
                probs=(0.9, 0.05, 0.05),
                reason="Need more evidence.",
            ),
            GovernanceDecision(
                mode=AnswerMode.TRUSTWORTHY,
                probs=(0.05, 0.05, 0.9),
                reason="Enough evidence.",
            ),
        ]
    )

    decisions = pyrrho.decide_many(
        "What happened?",
        [
            [],
            [SimpleNamespace(content="No answer here.")],
            [
                SimpleNamespace(content="The release shipped."),
                SimpleNamespace(content="The changelog confirms it."),
            ],
        ],
    )

    assert [decision.mode for decision in decisions] == [
        AnswerMode.ABSTAIN,
        AnswerMode.ABSTAIN,
        AnswerMode.TRUSTWORTHY,
    ]
    pyrrho._predict_context_batches.assert_called_once_with(
        "What happened?",
        [
            ["No answer here."],
            ["The release shipped.", "The changelog confirms it."],
        ],
    )


def test_head_decision_threshold_falls_back_from_weak_trustworthy():
    """Weak TRUSTWORTHY predictions should expose the threshold fallback."""
    decision = _head_decision(
        torch.tensor([0.1, 0.2, 0.25]),
        {0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"},
        trustworthy_threshold=0.5,
    )

    assert decision.raw_label == "TRUSTWORTHY"
    assert decision.final_label == "DISPUTED"
    assert decision.used_threshold_fallback is True
    assert decision.threshold == 0.5
    assert decision.runner_up_label == "TRUSTWORTHY"
    assert decision.margin_to_runner_up < 0.0
    assert decision.probabilities.keys() == {"ABSTAIN", "DISPUTED", "TRUSTWORTHY"}


def test_decision_from_outputs_includes_optional_heads():
    """g4-style optional heads should survive conversion into GovernanceDecision."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._id2label = {0: "ABSTAIN", 1: "DISPUTED", 2: "TRUSTWORTHY"}
    pyrrho._query_contract_id2label = {0: "evidence_sufficiency", 1: "structured_lookup"}
    pyrrho._route_id2label = {0: "technology_computing", 1: "general_commonsense"}
    pyrrho._taxonomy_id2label = {0: "direct_answer", 1: "evidence_absent"}
    pyrrho._optional_id2labels = {
        "retrieval_action": {0: "answer_now", 1: "retrieve_more"},
        "gap_type": {0: "none", 1: "missing_specific_fact"},
        "answerability_shape": {0: "direct_answer", 1: "structured_reasoning"},
        "retrieval_modality": {0: "unstructured_text", 1: "structured_table"},
    }
    pyrrho._scalar_fields = ("evidence_sufficiency", "evidence_failure_severity")
    pyrrho._trustworthy_threshold = 0.44
    outputs = {
        "governance_logits": torch.tensor([[0.0, 0.1, 2.0]]),
        "query_contract_logits": torch.tensor([[0.0, 1.0]]),
        "route_logits": torch.tensor([[1.0, 0.0]]),
        "taxonomy_logits": torch.tensor([[1.0, 0.0]]),
        "retrieval_action_logits": torch.tensor([[0.0, 2.0]]),
        "gap_type_logits": torch.tensor([[0.0, 2.0]]),
        "answerability_shape_logits": torch.tensor([[0.0, 2.0]]),
        "retrieval_modality_logits": torch.tensor([[0.0, 2.0]]),
        "scalar_preds": torch.tensor([[0.75, 0.25]]),
    }

    decision = _decision_from_outputs(pyrrho, outputs, 0)

    assert decision.mode is AnswerMode.TRUSTWORTHY
    assert decision.governance.threshold == 0.44
    assert decision.query_contract.final_label == "structured_lookup"
    assert decision.retrieval_action.final_label == "retrieve_more"
    assert decision.gap_type.final_label == "missing_specific_fact"
    assert decision.answerability_shape.final_label == "structured_reasoning"
    assert decision.retrieval_modality.final_label == "structured_table"
    assert decision.heads["retrieval_action"] is decision.retrieval_action
    assert decision.scalars == {
        "evidence_sufficiency": 0.75,
        "evidence_failure_severity": 0.25,
    }


def test_predict_query_includes_query_only_heads():
    """classify_query should return the full pre-retrieval Pyrrho query shape."""
    pyrrho = Pyrrho.__new__(Pyrrho)
    pyrrho._load = MagicMock()
    pyrrho._run_batch = MagicMock(
        return_value={
            "query_contract_logits": torch.tensor([[0.0, 2.0]]),
            "route_logits": torch.tensor([[2.0, 0.0]]),
            "answerability_shape_logits": torch.tensor([[0.0, 2.0]]),
            "retrieval_modality_logits": torch.tensor([[0.0, 2.0]]),
            "retrieval_action_logits": torch.tensor([[0.0, 2.0]]),
        }
    )
    pyrrho._query_contract_id2label = {0: "evidence_sufficiency", 1: "structured_lookup"}
    pyrrho._route_id2label = {0: "technology_computing", 1: "general_commonsense"}
    pyrrho._optional_id2labels = {
        "retrieval_action": {0: "answer_now", 1: "retrieve_more"},
        "answerability_shape": {0: "direct_answer", 1: "structured_reasoning"},
        "retrieval_modality": {0: "unstructured_text", 1: "structured_table"},
    }

    decision = pyrrho.classify_query("Which benchmark row is lower?")

    assert isinstance(decision, QueryDecision)
    assert decision.query_contract.final_label == "structured_lookup"
    assert decision.route.final_label == "technology_computing"
    assert decision.answerability_shape.final_label == "structured_reasoning"
    assert decision.retrieval_modality.final_label == "structured_table"
    assert decision.heads["query_contract"] is decision.query_contract
    assert "retrieval_action" not in decision.heads


def test_load_trustworthy_threshold_reads_manifest(tmp_path):
    """Packaged release thresholds should override the legacy constant."""
    (tmp_path / "manifest.json").write_text(
        '{"release": {"trustworthy_threshold": 0.44}}',
        encoding="utf-8",
    )

    assert _load_trustworthy_threshold(tmp_path) == 0.44

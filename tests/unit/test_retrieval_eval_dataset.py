# tests/unit/test_retrieval_eval_dataset.py
"""Tests for retrieval benchmark dataset loading."""

from __future__ import annotations

import json
from types import SimpleNamespace

from tools.retrieval_eval.dataset import load_dataset
from tools.retrieval_eval.matchers import rank_units


def test_load_dataset_materializes_repo_file_subset(tmp_path):
    """corpus_files datasets copy selected repo files into a temporary corpus."""
    dataset_path = tmp_path / "query_profile_unit.json"
    dataset_path.write_text(
        json.dumps(
            {
                "mode": "query_profile",
                "collection": "unit_query_profile_dataset",
                "corpus_files": ["fitz_sage/integrations/pyrrho.py"],
                "queries": [
                    {
                        "id": "qp-01",
                        "query": "Where does Pyrrho classify query signals?",
                        "relevant": [{"grade": 2, "path": "fitz_sage/integrations/pyrrho.py"}],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    dataset = load_dataset(dataset_path)

    assert dataset.mode == "query_profile"
    assert dataset.collection == "unit_query_profile_dataset"
    assert dataset.corpus_files == ("fitz_sage/integrations/pyrrho.py",)
    assert (dataset.corpus / "fitz_sage/integrations/pyrrho.py").is_file()


def test_query_profile_mode_matches_code_paths():
    """query_profile eval units use the same path matcher as code retrieval."""
    result = SimpleNamespace(
        file_path="C:/tmp/corpus/fitz_sage/engines/fitz_krag/retrieval_profile.py",
        metadata={},
        address=None,
        content="",
    )

    units = rank_units(
        "query_profile",
        [result],
        [{"grade": 2, "path": "fitz_sage/engines/fitz_krag/retrieval_profile.py"}],
    )

    assert units[0].rank == 1

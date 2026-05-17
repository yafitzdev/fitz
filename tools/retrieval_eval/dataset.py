# tools/retrieval_eval/dataset.py
"""Unified retrieval-benchmark dataset — one schema across code/section/table.

A dataset file lives at ``datasets/<mode>.json``:

    {
      "mode": "section",
      "corpus": "tools/retrieval_eval/corpus/section",   # repo-relative
      "collection": "reval_section",
      "queries": [
        {
          "id": "section-01",
          "query": "What are the three paradigms of RAG?",
          "vocab_mismatch": false,
          "relevant": [
            {"grade": 2, "doc": "rag_survey.pdf", "heading": "Naive RAG", "page": 5},
            {"grade": 1, "doc": "rag_survey.pdf", "heading": "Overview of RAG"}
          ]
        }
      ]
    }

``grade``: 2 = critical (must retrieve), 1 = relevant (should retrieve).
Locator fields per mode — what identifies a relevant unit:
  code    -> path
  section -> doc, heading, [page]
  table   -> doc, value
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

MODES = ("code", "section", "table")

# Fields that must be present on every unit, by mode.
_LOCATORS = {"code": ("path",), "section": ("doc", "heading"), "table": ("doc", "value")}

_REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class BenchQuery:
    """One benchmark query and its curated ground truth."""

    id: str
    query: str
    relevant: list[dict]
    vocab_mismatch: bool = False


@dataclass(frozen=True)
class Dataset:
    """A loaded, validated dataset for one retrieval mode."""

    mode: str
    corpus: Path
    collection: str
    queries: list[BenchQuery]


def load_dataset(path: str | Path) -> Dataset:
    """Load and validate one dataset file.

    Raises:
        ValueError: on an unknown mode, a missing corpus, a duplicate query id,
            or a unit missing a required locator field.
    """
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    mode = raw.get("mode")
    if mode not in MODES:
        raise ValueError(f"{path.name}: mode must be one of {MODES}, got {mode!r}")

    corpus = (_REPO_ROOT / raw["corpus"]).resolve()
    if not corpus.exists():
        raise ValueError(f"{path.name}: corpus path does not exist: {corpus}")

    queries: list[BenchQuery] = []
    seen: set[str] = set()
    for q in raw.get("queries", []):
        qid = q["id"]
        if qid in seen:
            raise ValueError(f"{path.name}: duplicate query id {qid!r}")
        seen.add(qid)
        _validate_units(path, mode, qid, q["relevant"])
        queries.append(
            BenchQuery(
                id=qid,
                query=q["query"],
                relevant=q["relevant"],
                vocab_mismatch=bool(q.get("vocab_mismatch", False)),
            )
        )

    if not queries:
        raise ValueError(f"{path.name}: dataset has no queries")

    return Dataset(
        mode=mode,
        corpus=corpus,
        collection=raw["collection"],
        queries=queries,
    )


def _validate_units(path: Path, mode: str, qid: str, units: list[dict]) -> None:
    if not units:
        raise ValueError(f"{path.name}: query {qid!r} has no relevant units")
    for unit in units:
        if unit.get("grade") not in (1, 2):
            raise ValueError(f"{path.name}: query {qid!r} unit grade must be 1 or 2")
        for locator in _LOCATORS[mode]:
            if not unit.get(locator):
                raise ValueError(
                    f"{path.name}: query {qid!r} unit missing {locator!r} for mode {mode!r}"
                )

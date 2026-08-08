"""Freeze a score-blind development/holdout split for EnterpriseRAG-Bench."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.enterprise_rag import (
    SPEC,
    EnterpriseQuestion,
    EnterpriseRagSpec,
    load_questions,
)

SCHEMA_VERSION = 1
DEFAULT_SEED = 20260731
SELECTION_ALGORITHM = "category-stratified-sha256-30-percent-half-up-v1"
SCORED_SPLITS = ("development", "holdout")
ALL_SELECTIONS = (*SCORED_SPLITS, "all-scored")


def build_split_manifest(
    questions: dict[str, EnterpriseQuestion],
    *,
    spec: EnterpriseRagSpec = SPEC,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Create deterministic category-stratified IDs without using retrieval scores."""
    scored_by_category: dict[str, list[str]] = {}
    unscored: list[str] = []
    for question_id, question in questions.items():
        if question.expected_document_ids:
            scored_by_category.setdefault(question.question_type, []).append(question_id)
        else:
            unscored.append(question_id)

    development: list[str] = []
    holdout: list[str] = []
    for category in sorted(scored_by_category):
        ordered = sorted(
            scored_by_category[category],
            key=lambda query_id: (_selection_hash(seed, category, query_id), query_id),
        )
        development_count = (len(ordered) * 3 + 5) // 10
        development.extend(ordered[:development_count])
        holdout.extend(ordered[development_count:])

    return {
        "schema_version": SCHEMA_VERSION,
        "name": "enterprise-rag-bench-retrieval-split-v1",
        "purpose": (
            "Freeze a category-stratified development/holdout split before any "
            "Fitz-Sage retrieval result is inspected."
        ),
        "source": {
            "release": spec.release,
            "archive_sha256": spec.archive_sha256,
            "questions_sha256": spec.questions_sha256,
            "questions": len(questions),
        },
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "development_fraction": "3/10",
            "rounding": "nearest integer, halves upward, independently per category",
            "inputs": ["question_id", "question_type", "has_expected_documents"],
            "retrieval_scores_used": False,
        },
        "splits": {
            "development": _split_record(development, questions),
            "holdout": _split_record(holdout, questions),
            "unscored": _split_record(sorted(unscored), questions),
        },
    }


def load_split_manifest(
    path: Path,
    questions: dict[str, EnterpriseQuestion],
    *,
    spec: EnterpriseRagSpec = SPEC,
) -> dict[str, Any]:
    """Load and fully validate a frozen split against the pinned questions."""
    source = Path(path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid EnterpriseRAG-Bench split JSON: {source}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported EnterpriseRAG-Bench split schema: {source}")
    if manifest.get("selection", {}).get("algorithm") != SELECTION_ALGORITHM:
        raise ValueError(f"Unexpected split selection algorithm: {source}")
    expected_source = {
        "release": spec.release,
        "archive_sha256": spec.archive_sha256,
        "questions_sha256": spec.questions_sha256,
        "questions": len(questions),
    }
    if manifest.get("source") != expected_source:
        raise ValueError(f"Split source identity does not match pinned release: {source}")
    if manifest != build_split_manifest(
        questions,
        spec=spec,
        seed=int(manifest["selection"]["seed"]),
    ):
        raise ValueError(f"Split content is not reproducible from pinned questions: {source}")
    return manifest


def split_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical SHA-256 identity of a split manifest."""
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def selection_from_manifest(
    manifest: dict[str, Any],
    questions: dict[str, EnterpriseQuestion],
    selection: str,
) -> tuple[tuple[str, ...], dict[str, dict[str, Any]], dict[str, Any]]:
    """Resolve ordered scored IDs, grouping metadata, and report metadata."""
    if selection not in ALL_SELECTIONS:
        raise ValueError(f"Unknown EnterpriseRAG-Bench selection: {selection}")
    if selection == "all-scored":
        query_ids = tuple(manifest["splits"][split]["query_ids"] for split in SCORED_SPLITS)
        flattened = tuple(query_id for values in query_ids for query_id in values)
    else:
        flattened = tuple(manifest["splits"][selection]["query_ids"])
    metadata = {
        query_id: {
            "category": questions[query_id].question_type,
            "source_types": list(questions[query_id].source_types),
            "expected_documents": len(set(questions[query_id].expected_document_ids)),
            "expected_document_entries": len(questions[query_id].expected_document_ids),
        }
        for query_id in flattened
    }
    return (
        flattened,
        metadata,
        {
            "split": selection,
            "split_manifest_name": manifest["name"],
            "split_manifest_sha256": split_manifest_digest(manifest),
            "category_counts": dict(
                sorted(Counter(value["category"] for value in metadata.values()).items())
            ),
            "unscored_queries": manifest["splits"]["unscored"]["queries"],
        },
    )


def write_frozen_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Create an immutable manifest, accepting an identical existing file."""
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    if output.exists():
        if output.read_text(encoding="utf-8") != rendered:
            raise FileExistsError(f"Frozen split already has different content: {output}")
        return
    output.write_text(rendered, encoding="utf-8", newline="\n")


def _split_record(
    query_ids: list[str],
    questions: dict[str, EnterpriseQuestion],
) -> dict[str, Any]:
    return {
        "queries": len(query_ids),
        "category_counts": dict(
            sorted(Counter(questions[query_id].question_type for query_id in query_ids).items())
        ),
        "query_ids": query_ids,
    }


def _selection_hash(seed: int, category: str, query_id: str) -> str:
    return hashlib.sha256(f"{seed}:{category}:{query_id}".encode("utf-8")).hexdigest()


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path(".benchmark-data/enterprise-rag-bench/v1.0.0/questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fixtures/enterprise_rag_split_v1.json"),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    questions = load_questions(args.questions)
    manifest = build_split_manifest(questions, seed=args.seed)
    write_frozen_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": split_manifest_digest(manifest),
                "counts": {name: value["queries"] for name, value in manifest["splits"].items()},
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Freeze a label-derived BEIR holdout for semantic query expansion."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.beir import (
    DATASETS,
    PreparedDataset,
    iter_corpus,
    load_qrels,
    load_queries,
    prepare_dataset,
    projected_content,
)
from benchmarks.fitz_bench.external_data import file_digest
from benchmarks.fitz_bench.retrieval_eval import tokenize

SCHEMA_VERSION = 1
SELECTION_ALGORITHM = "positive-max-token-jaccard-tertiles-sha256-v1"
DEFAULT_DATASETS = ("arguana", "quora")
DEFAULT_SAMPLE_SIZE = 120
DEFAULT_SEED = 20260730
GROUPS = ("low", "medium", "high")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    prepared = [
        prepare_dataset(
            args.cache_dir,
            dataset,
            max_download_bytes=int(args.max_download_gib * 1024**3),
            max_extracted_bytes=int(args.max_extracted_gib * 1024**3),
            offline=args.offline,
            progress=print,
        )
        for dataset in args.datasets
    ]
    manifest = build_holdout_manifest(
        prepared,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    _write_frozen_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": query_manifest_digest(manifest),
                "datasets": {
                    name: value["group_counts"] for name, value in manifest["datasets"].items()
                },
            },
            indent=2,
        )
    )
    return 0


def build_holdout_manifest(
    datasets: list[PreparedDataset],
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    """Select equal low, medium, and high lexical-overlap query groups."""
    if sample_size < len(GROUPS) or sample_size % len(GROUPS):
        raise ValueError(f"sample_size must be a positive multiple of {len(GROUPS)}")
    if not datasets:
        raise ValueError("At least one prepared dataset is required.")

    selected = {
        dataset.name: _select_dataset_queries(
            dataset,
            sample_size=sample_size,
            seed=seed,
        )
        for dataset in datasets
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "name": "beir-semantic-vocabulary-holdout-v1",
        "purpose": (
            "Measure managed semantic query expansion across frozen lexical-overlap "
            "strata without using retrieval scores for selection."
        ),
        "selection": {
            "algorithm": SELECTION_ALGORITHM,
            "seed": seed,
            "sample_size_per_dataset": sample_size,
            "queries_per_group": sample_size // len(GROUPS),
            "groups": list(GROUPS),
            "tokenizer": "benchmark Unicode word tokens, casefolded",
            "positive_overlap": "maximum token-set Jaccard over judged-relevant documents",
            "self_document_policy": "exclude corpus document whose ID equals the query ID",
            "missing_positive_policy": (
                "record missing positive query/document pairs and exclude only queries "
                "with no available judged-relevant document"
            ),
        },
        "datasets": selected,
    }


def load_query_manifest(path: Path) -> dict[str, Any]:
    """Load and validate a frozen query manifest."""
    source = Path(path).resolve()
    try:
        manifest = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid query manifest JSON: {source}") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported query manifest schema: {source}")
    if manifest.get("selection", {}).get("algorithm") != SELECTION_ALGORITHM:
        raise ValueError(f"Unexpected query manifest selection algorithm: {source}")
    datasets = manifest.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError(f"Query manifest has no datasets: {source}")
    for name, value in datasets.items():
        if not isinstance(name, str) or not isinstance(value, dict):
            raise TypeError(f"Invalid query manifest dataset entry: {name!r}")
        queries = value.get("queries")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Query manifest dataset {name!r} has no queries")
        seen: set[str] = set()
        for item in queries:
            if not isinstance(item, dict):
                raise TypeError(f"Invalid query manifest record for {name!r}")
            query_id = item.get("query_id")
            group = item.get("group")
            if not isinstance(query_id, str) or not query_id or query_id in seen:
                raise ValueError(f"Invalid or duplicate query ID for {name!r}: {query_id!r}")
            if group not in GROUPS:
                raise ValueError(f"Invalid query group for {name}/{query_id}: {group!r}")
            seen.add(query_id)
    return manifest


def query_manifest_digest(manifest: dict[str, Any]) -> str:
    """Return the canonical SHA-256 identity of a query manifest."""
    payload = json.dumps(
        manifest,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def manifest_dataset_names(manifest: dict[str, Any]) -> list[str]:
    """Return frozen dataset names in manifest order."""
    return list(manifest["datasets"])


def manifest_query_selection(
    manifest: dict[str, Any],
    dataset: PreparedDataset,
    *,
    available_query_ids: set[str],
) -> tuple[list[str], dict[str, dict[str, Any]], dict[str, Any]]:
    """Resolve one dataset's frozen query IDs and per-query metadata."""
    raw = manifest["datasets"].get(dataset.name)
    if not isinstance(raw, dict):
        raise KeyError(f"Query manifest does not contain dataset {dataset.name!r}")
    if raw.get("archive_md5") != dataset.md5:
        raise ValueError(
            f"Query manifest archive mismatch for {dataset.name}: "
            f"{raw.get('archive_md5')!r} != {dataset.md5!r}"
        )
    records = raw["queries"]
    query_ids = [str(item["query_id"]) for item in records]
    missing = [query_id for query_id in query_ids if query_id not in available_query_ids]
    if missing:
        raise ValueError(f"Query manifest references missing {dataset.name} queries: {missing[:3]}")
    metadata = {
        str(item["query_id"]): {
            "group": str(item["group"]),
            "positive_max_token_jaccard": float(item["positive_max_token_jaccard"]),
        }
        for item in records
    }
    summary = {
        "manifest_name": manifest["name"],
        "manifest_sha256": query_manifest_digest(manifest),
        "selection_algorithm": manifest["selection"]["algorithm"],
        "group_counts": dict(Counter(item["group"] for item in records)),
    }
    return query_ids, metadata, summary


def _select_dataset_queries(
    dataset: PreparedDataset,
    *,
    sample_size: int,
    seed: int,
) -> dict[str, Any]:
    queries = load_queries(Path(dataset.source_queries))
    qrels = load_qrels(Path(dataset.source_qrels))
    positive_ids = {
        document_id
        for query_id, judgments in qrels.items()
        for document_id, score in judgments.items()
        if score > 0 and document_id != query_id
    }
    positive_documents: dict[str, str] = {}
    for record in iter_corpus(Path(dataset.source_corpus)):
        document_id = str(record["_id"])
        if document_id in positive_ids:
            positive_documents[document_id] = projected_content(record)
    missing_document_ids = positive_ids - set(positive_documents)
    missing_positive_qrels = sorted(
        (
            {"query_id": query_id, "document_id": document_id}
            for query_id, judgments in qrels.items()
            for document_id, score in judgments.items()
            if score > 0 and document_id != query_id and document_id in missing_document_ids
        ),
        key=lambda item: (item["query_id"], item["document_id"]),
    )

    scored: list[tuple[str, float]] = []
    queries_without_available_positives: list[str] = []
    for query_id, judgments in qrels.items():
        if query_id not in queries:
            raise ValueError(f"{dataset.name} qrels reference missing query {query_id!r}")
        relevant = [
            document_id
            for document_id, score in judgments.items()
            if score > 0 and document_id != query_id and document_id in positive_documents
        ]
        if not relevant:
            queries_without_available_positives.append(query_id)
            continue
        query_tokens = set(tokenize(queries[query_id]))
        overlap = max(
            _jaccard(query_tokens, set(tokenize(positive_documents[document_id])))
            for document_id in relevant
        )
        scored.append((query_id, overlap))
    if len(scored) < sample_size:
        raise ValueError(
            f"{dataset.name} has {len(scored)} eligible queries; cannot select {sample_size}"
        )

    ordered = sorted(scored, key=lambda item: (item[1], item[0]))
    grouped: dict[str, list[tuple[str, float]]] = {group: [] for group in GROUPS}
    for index, item in enumerate(ordered):
        group_index = min(len(GROUPS) - 1, index * len(GROUPS) // len(ordered))
        grouped[GROUPS[group_index]].append(item)

    per_group = sample_size // len(GROUPS)
    selected: list[dict[str, Any]] = []
    group_ranges: dict[str, dict[str, float]] = {}
    for group in GROUPS:
        candidates = grouped[group]
        if len(candidates) < per_group:
            raise ValueError(
                f"{dataset.name}/{group} has {len(candidates)} candidates; "
                f"cannot select {per_group}"
            )
        chosen = sorted(
            candidates,
            key=lambda item: (
                _selection_hash(seed, dataset.name, group, item[0]),
                item[0],
            ),
        )[:per_group]
        selected.extend(
            {
                "query_id": query_id,
                "group": group,
                "positive_max_token_jaccard": round(overlap, 8),
            }
            for query_id, overlap in chosen
        )
        group_ranges[group] = {
            "candidate_min": min(overlap for _query_id, overlap in candidates),
            "candidate_max": max(overlap for _query_id, overlap in candidates),
            "selected_mean": sum(overlap for _query_id, overlap in chosen) / len(chosen),
        }

    return {
        "archive_md5": dataset.md5,
        "corpus_sha256": file_digest(Path(dataset.source_corpus), "sha256"),
        "queries_sha256": file_digest(Path(dataset.source_queries), "sha256"),
        "qrels_sha256": file_digest(Path(dataset.source_qrels), "sha256"),
        "corpus_documents": dataset.corpus_documents,
        "available_judged_queries": len(qrels),
        "eligible_queries": len(scored),
        "missing_positive_qrels": missing_positive_qrels,
        "queries_without_available_positives": sorted(queries_without_available_positives),
        "ignore_identical_ids": dataset.ignore_identical_ids,
        "group_counts": dict(Counter(item["group"] for item in selected)),
        "group_ranges": group_ranges,
        "queries": selected,
    }


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0


def _selection_hash(seed: int, dataset: str, group: str, query_id: str) -> str:
    payload = f"{seed}:{dataset}:{group}:{query_id}".encode()
    return hashlib.sha256(payload).hexdigest()


def _write_frozen_manifest(path: Path, manifest: dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(manifest, indent=2, ensure_ascii=True) + "\n"
    if output.exists():
        existing = load_query_manifest(output)
        if existing != manifest:
            raise FileExistsError(
                f"Frozen query manifest already exists with different content: {output}"
            )
        return
    output.write_text(rendered, encoding="utf-8", newline="\n")


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to sample. Repeat; defaults to ArguAna and Quora.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark-data/beir"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/fixtures/beir_semantic_holdout_v1.json"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parsed = parser.parse_args(argv)
    parsed.datasets = parsed.datasets or list(DEFAULT_DATASETS)
    if len(set(parsed.datasets)) != len(parsed.datasets):
        parser.error("--dataset values must be unique")
    if parsed.sample_size < len(GROUPS) or parsed.sample_size % len(GROUPS):
        parser.error(f"--sample-size must be a positive multiple of {len(GROUPS)}")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

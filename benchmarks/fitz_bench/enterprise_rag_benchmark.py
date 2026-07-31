"""Run frozen EnterpriseRAG-Bench retrieval evaluation through Fitz-Sage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.enterprise_rag import (
    SPEC,
    iter_archive_documents,
    load_questions,
    prepare_dataset,
    queries_and_qrels,
)
from benchmarks.fitz_bench.enterprise_rag_split import (
    ALL_SELECTIONS,
    load_split_manifest,
    selection_from_manifest,
    split_manifest_digest,
)
from benchmarks.fitz_bench.external_retrieval import (
    ExternalQuerySelection,
    ExternalRetrievalDataset,
    evaluate_external_dataset,
)
from benchmarks.fitz_bench.retrieval_ablation import (
    ablation_names,
    get_ablation,
)
from benchmarks.fitz_bench.retrieval_eval import (
    aggregate_metrics,
    metric_formulas,
    summarize_latency,
)
from benchmarks.fitz_bench.sqlite_bm25 import SqliteBm25

DEFAULT_SPLIT_MANIFEST = Path("benchmarks/fixtures/enterprise_rag_split_v1.json")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    prepared = prepare_dataset(
        args.cache_dir,
        max_download_bytes=int(args.max_download_gib * 1024**3),
        max_extracted_bytes=int(args.max_extracted_gib * 1024**3),
        offline=args.offline,
        progress=print,
    )
    questions = load_questions(prepared.questions_path)
    split_manifest = load_split_manifest(args.split_manifest, questions)
    manifest_sha256 = split_manifest_digest(split_manifest)
    args.split_manifest_sha256 = manifest_sha256

    if args.prepare_only:
        report = {
            "run": _run_metadata(args, run_id, started),
            "dataset": prepared.as_dict(),
            "split_manifest": _split_manifest_report(args, split_manifest),
            "gate": {"type": "operational_only", "passed": True, "failures": []},
            "complete": True,
        }
        return _write_report(report, args.output, args.markdown)

    queries, qrels = queries_and_qrels(questions)
    query_ids, metadata, selection_report = selection_from_manifest(
        split_manifest,
        questions,
        args.selection,
    )
    selection_identity = "\n".join(query_ids)
    selection_digest = hashlib.sha256(
        f"{manifest_sha256}:{args.selection}:{selection_identity}".encode("utf-8")
    ).hexdigest()
    baseline_path = Path(args.baseline_db).resolve()
    baseline_fingerprint = {
        **prepared.fingerprint(),
        "baseline_schema": "sqlite-fts5-unicode61-v1",
    }
    dataset = ExternalRetrievalDataset(
        name=prepared.name,
        report={
            **prepared.as_dict(),
            "questions": len(questions),
            "scored_queries": len(qrels),
            "unscored_queries": len(questions) - len(qrels),
            "raw_relevance_entries": sum(
                len(question.expected_document_ids) for question in questions.values()
            ),
            "unique_relevance_links": sum(len(values) for values in qrels.values()),
        },
        fingerprint=prepared.fingerprint(),
        corpus_dir=prepared.corpus_dir,
        mapping_path=prepared.mapping_path,
        corpus_documents=prepared.spec.corpus_documents,
        adapter_schema_version=prepared.adapter_schema_version,
        queries=queries,
        qrels=qrels,
        baseline_factory=lambda: SqliteBm25.open_or_build(
            baseline_path,
            fingerprint=baseline_fingerprint,
            expected_documents=prepared.spec.corpus_documents,
            documents=lambda: iter_archive_documents(prepared),
            progress=print,
        ),
        baseline_report={
            "implementation": "benchmark-local SQLite FTS5 BM25",
            "analyzer": "SQLite unicode61",
            "k1": 1.2,
            "b": 0.75,
            "database": str(baseline_path),
            "corpus_source": "verified official ZIP bytes",
        },
        allow_duplicate_document_ids=True,
    )
    selection = ExternalQuerySelection(
        name=args.selection,
        query_ids=query_ids,
        metadata=metadata,
        report={
            **selection_report,
            "split_manifest": _split_manifest_report(args, split_manifest),
        },
        digest=selection_digest,
    )
    ablation = get_ablation(args.ablation) if args.ablation else None
    result = evaluate_external_dataset(
        dataset,
        selection,
        workspace_root=args.workspace_root,
        run_id=run_id,
        namespace="enterprise_rag",
        cutoffs=args.cutoffs,
        index_mode=args.index_mode,
        governance=args.governance,
        baseline_only=args.baseline_only,
        reuse_workspace=args.reuse_workspace,
        reuse_index=args.reuse_index,
        resume_queries=args.resume_queries,
        evaluation_paths=(
            Path(__file__),
            Path(__file__).with_name("enterprise_rag.py"),
            Path(__file__).with_name("enterprise_rag_split.py"),
            Path(__file__).with_name("sqlite_bm25.py"),
            Path(__file__).with_name("retrieval_eval.py"),
            Path(__file__).with_name("retrieval_ablation.py"),
            args.split_manifest,
        ),
        ablation=ablation,
    )
    result["grouped_results"] = _grouped_results(result.get("records", []))
    failures = _operational_failures(result)
    report = {
        "run": _run_metadata(args, run_id, started),
        "metric_formulas": metric_formulas(),
        "evaluation_contract": {
            "primary_metric": f"Recall@{max(args.cutoffs)}",
            "secondary_metrics": [
                f"NDCG@{_ranking_cutoff(args.cutoffs)}",
                f"MRR@{_ranking_cutoff(args.cutoffs)}",
            ],
            "scored_queries": "Only questions with at least one expected document.",
            "unscored_queries": (
                "High-level and information-not-found questions are reported but not "
                "assigned retrieval relevance scores."
            ),
            "duplicate_identity_policy": (
                "Preserve every physical source file, map both files to their official "
                "document ID, and deduplicate ranked official IDs before scoring."
            ),
            "pyrrho_policy": (
                "Store exact Pyrrho output for diagnostics; do not alter or score its "
                "governance decision in Fitz-Sage."
            ),
            "answer_generation_evaluated": False,
        },
        "dataset": result,
        "gate": {
            "type": "operational_and_integrity_only",
            "passed": not failures,
            "failures": failures,
        },
        "complete": True,
    }
    exit_code = _write_report(report, args.output, args.markdown)
    return exit_code if not failures else 1


def _grouped_results(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "category": _summarize_groups(
            records,
            {
                str(record["evaluation"]["category"]): True
                for record in records
                if "evaluation" in record
            },
            membership=lambda record, group: record["evaluation"]["category"] == group,
        ),
        "source_type": _summarize_groups(
            records,
            {
                str(source_type): True
                for record in records
                for source_type in record.get("evaluation", {}).get("source_types", [])
            },
            membership=lambda record, group: group
            in record.get("evaluation", {}).get("source_types", []),
        ),
        "document_cardinality": _summarize_groups(
            records,
            {"single": True, "multiple": True},
            membership=lambda record, group: (
                int(record["evaluation"]["expected_documents"]) == 1
                if group == "single"
                else int(record["evaluation"]["expected_documents"]) > 1
            ),
        ),
    }


def _summarize_groups(
    records: list[dict[str, Any]],
    groups: dict[str, bool],
    *,
    membership: Any,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for group in sorted(groups):
        selected = [record for record in records if membership(record, group)]
        stages = sorted({stage for record in selected for stage in record.get("metrics", {})})
        output[group] = {
            "queries": len(selected),
            "metrics": {
                stage: aggregate_metrics(
                    record["metrics"][stage]
                    for record in selected
                    if stage in record.get("metrics", {})
                )
                for stage in stages
            },
            "fitz_sage_latency": summarize_latency(
                [
                    float(record["latency_seconds"]["fitz_sage"])
                    for record in selected
                    if "fitz_sage" in record.get("latency_seconds", {})
                ]
            ),
            "failure_attribution": dict(
                sorted(
                    Counter(
                        str(record["failure_attribution"])
                        for record in selected
                        if "failure_attribution" in record
                    ).items()
                )
            ),
        }
    return output


def _operational_failures(result: dict[str, Any]) -> list[str]:
    ingestion = result.get("ingestion")
    failures: list[str] = []
    if isinstance(ingestion, dict) and ingestion.get("failures"):
        failures.append("corpus ingestion contains failed or unsupported documents")
    mapping = ingestion.get("mapping", {}) if isinstance(ingestion, dict) else {}
    if mapping and not mapping.get("complete"):
        failures.append("source-to-official-ID mapping is incomplete")
    return failures


def _split_manifest_report(
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    return {
        "path": str(args.split_manifest.resolve()),
        "name": manifest["name"],
        "sha256": split_manifest_digest(manifest),
        "selection_algorithm": manifest["selection"]["algorithm"],
        "retrieval_scores_used_for_selection": False,
    }


def _run_metadata(
    args: argparse.Namespace,
    run_id: str,
    started: float,
) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "git": _git_state(),
        "dataset_release": SPEC.release,
        "selection": args.selection,
        "split_manifest": {
            "path": str(args.split_manifest.resolve()),
            "sha256": getattr(args, "split_manifest_sha256", None),
        },
        "index_mode": args.index_mode,
        "baseline_only": args.baseline_only,
        "cutoffs": args.cutoffs,
        "governance_override": args.governance,
        "ablation": get_ablation(args.ablation).as_dict() if args.ablation else None,
        "reuse_workspace": args.reuse_workspace,
        "reuse_index": args.reuse_index,
        "resume_queries": args.resume_queries,
        "duration_seconds": time.perf_counter() - started,
    }


def _git_state() -> dict[str, Any]:
    def command(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            check=False,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else ""

    return {
        "commit": command("rev-parse", "HEAD") or None,
        "branch": command("branch", "--show-current") or None,
        "dirty": bool(command("status", "--porcelain")),
    }


def _write_report(report: dict[str, Any], output: Path, markdown: Path | None) -> int:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if markdown is not None:
        markdown = Path(markdown).resolve()
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(output), "gate": report["gate"]}, indent=2))
    return 0


def _markdown(report: dict[str, Any]) -> str:
    run = report["run"]
    lines = [
        "# EnterpriseRAG-Bench Retrieval Evaluation",
        "",
        f"- Run: `{run['run_id']}`",
        f"- Release: `{run['dataset_release']}`",
        f"- Selection: `{run['selection']}`",
        f"- Gate: `{'PASS' if report['gate']['passed'] else 'FAIL'}`",
        f"- Split manifest: `{run['split_manifest']['sha256']}`",
        "",
    ]
    result = report.get("dataset", {})
    summary = result.get("summary", {})
    metrics = summary.get("metrics", {})
    if metrics:
        cutoffs = result["selection"]["cutoffs"]
        ranking_cutoff = _ranking_cutoff(cutoffs)
        selected_metrics = [
            f"NDCG@{ranking_cutoff}",
            f"Recall@{max(cutoffs)}",
            f"MRR@{ranking_cutoff}",
        ]
        lines.extend(
            [
                "## Overall",
                "",
                "| Stage | " + " | ".join(selected_metrics) + " |",
                "| --- | " + " | ".join("---:" for _ in selected_metrics) + " |",
            ]
        )
        for stage, values in metrics.items():
            cells = [f"{values.get(metric, 0.0):.4f}" for metric in selected_metrics]
            lines.append(f"| {stage} | " + " | ".join(cells) + " |")
        lines.append("")
    category = result.get("grouped_results", {}).get("category", {})
    if category:
        cutoff = max(result["selection"]["cutoffs"])
        lines.extend(
            [
                "## Categories",
                "",
                f"| Category | Queries | Baseline Recall@{cutoff} | Final Recall@{cutoff} |",
                "| --- | ---: | ---: | ---: |",
            ]
        )
        for name, values in category.items():
            grouped_metrics = values["metrics"]
            baseline = grouped_metrics.get("baseline", {}).get(f"Recall@{cutoff}", 0.0)
            final = grouped_metrics.get("final", {}).get(f"Recall@{cutoff}", 0.0)
            lines.append(f"| {name} | {values['queries']} | {baseline:.4f} | {final:.4f} |")
        lines.append("")
    lines.extend(
        [
            "## Scope",
            "",
            "This report scores document retrieval only. Questions without gold documents "
            "and answer generation are explicitly outside the score.",
            "",
        ]
    )
    return "\n".join(lines)


def _ranking_cutoff(cutoffs: list[int]) -> int:
    return max((value for value in cutoffs if value <= 10), default=min(cutoffs))


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".benchmark-data/enterprise-rag-bench"),
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(".bench_workspace/enterprise-rag-bench"),
    )
    parser.add_argument(
        "--baseline-db",
        type=Path,
        default=Path(".bench_workspace/enterprise-rag-bench/plain-bm25.sqlite3"),
    )
    parser.add_argument("--split-manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--selection", choices=ALL_SELECTIONS, default="development")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/enterprise_rag_latest.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmarks/results/enterprise_rag_latest.md"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--baseline-only", action="store_true")
    parser.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        type=int,
        help="Metric cutoff. Repeat; defaults to 1, 3, 5, 10, 20, and 50.",
    )
    parser.add_argument("--index-mode", choices=("source", "complete"), default="source")
    parser.add_argument("--ablation", choices=ablation_names())
    parser.add_argument("--governance")
    parser.add_argument(
        "--reuse-workspace",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--reuse-index",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--resume-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parsed = parser.parse_args(argv)
    parsed.cutoffs = sorted(set(parsed.cutoffs or [1, 3, 5, 10, 20, 50]))
    if any(value < 1 for value in parsed.cutoffs):
        parser.error("--cutoff must be positive")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    if parsed.resume_queries and not parsed.reuse_workspace:
        parser.error("--resume-queries requires --reuse-workspace")
    if parsed.reuse_index and not parsed.reuse_workspace:
        parser.error("--reuse-index requires --reuse-workspace")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

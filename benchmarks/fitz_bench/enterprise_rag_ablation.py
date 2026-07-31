"""Run paired EnterpriseRAG-Bench query-expansion and reranker ablations."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.enterprise_rag_split import ALL_SELECTIONS
from benchmarks.fitz_bench.paired_stats import derived_seed, paired_delta
from benchmarks.fitz_bench.retrieval_ablation import ablation_names, get_ablation

_EFFECTS = (
    ("expansion_without_reranker", "literal", "expansion"),
    ("reranker_without_expansion", "literal", "reranker"),
    ("expansion_with_reranker", "reranker", "full"),
    ("reranker_with_expansion", "expansion", "full"),
    ("full_vs_literal", "literal", "full"),
)
_EFFECT_LABELS = {
    "expansion_without_reranker": "Qwen, no reranker",
    "reranker_without_expansion": "Reranker, no Qwen",
    "expansion_with_reranker": "Qwen, reranker on",
    "reranker_with_expansion": "Reranker, Qwen on",
    "full_vs_literal": "Both vs literal",
}


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    run_dir = (args.runs_dir / f"{run_id}-{args.selection}").resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    variants = args.variants or list(ablation_names())
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}

    for index, variant in enumerate(variants, start=1):
        print(
            f"[{index}/{len(variants)}] Running EnterpriseRAG-Bench {variant} "
            f"({args.selection})",
            flush=True,
        )
        output = run_dir / f"{variant}.json"
        markdown = run_dir / f"{variant}.md"
        completed = subprocess.run(
            _variant_command(args, variant=variant, output=output, markdown=markdown),
            cwd=root,
            check=False,
        )
        if completed.returncode not in {0, 1} or not output.is_file():
            raise RuntimeError(
                f"EnterpriseRAG-Bench variant {variant!r} failed with exit code "
                f"{completed.returncode}; expected report {output}."
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not report.get("complete"):
            raise RuntimeError(f"Variant {variant!r} did not produce a complete report.")
        reports[variant] = report
        report_paths[variant] = output
        partial = build_ablation_report(
            reports,
            report_paths=report_paths,
            run_id=run_id,
            started=started,
            requested_variants=variants,
            bootstrap_samples=args.bootstrap_samples,
            seed=args.seed,
        )
        _write_json(args.output, partial)

    report = build_ablation_report(
        reports,
        report_paths=report_paths,
        run_id=run_id,
        started=started,
        requested_variants=variants,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    _write_json(args.output, report)
    _write_markdown(args.markdown, report)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "markdown": str(args.markdown.resolve()),
                "gate": report["gate"],
            },
            indent=2,
        )
    )
    return 0 if report["gate"]["passed"] else 1


def build_ablation_report(
    reports: dict[str, dict[str, Any]],
    *,
    report_paths: dict[str, Path],
    run_id: str,
    started: float,
    requested_variants: list[str],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Aggregate independent variants using paired query-level effects."""
    failures = _measurement_failures(reports, requested_variants)
    effects = _paired_effects(
        reports,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    categories = _category_effects(
        reports,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    complete = set(reports) == set(requested_variants)
    first = next(iter(reports.values()), {})
    first_result = first.get("dataset", {})
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "git": _shared_value(reports, lambda report: report["run"]["git"]),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "selection": first.get("run", {}).get("selection"),
            "split_manifest": first.get("run", {}).get("split_manifest"),
            "requested_variants": requested_variants,
            "completed_variants": list(reports),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "duration_seconds": time.perf_counter() - started,
        },
        "method": {
            "paired_by": "query_id",
            "confidence_interval": "deterministic paired percentile bootstrap, 95%",
            "index_reuse": "all variants query the same verified source collection",
            "baseline_reuse": "all variants query the same verified SQLite FTS5 index",
            "quality_gate": False,
            "interpretation": (
                "Fitz-Sage intentionally favors broad recall. Component deltas describe "
                "tradeoffs and are not optimization targets."
            ),
        },
        "dataset": first_result.get("dataset"),
        "cutoffs": first_result.get("selection", {}).get("cutoffs", []),
        "variants": {
            variant: {
                **get_ablation(variant).as_dict(),
                "report": str(report_paths[variant].resolve()),
                "source_gate": report.get("gate", {}),
                "metrics": report.get("dataset", {}).get("summary", {}).get("metrics", {}),
                "latency": report.get("dataset", {}).get("fitz_sage", {}).get("latency", {}),
                "timing": report.get("dataset", {}).get("summary", {}).get("timing", {}),
            }
            for variant, report in reports.items()
        },
        "effects": effects,
        "categories": categories,
        "gate": {
            "type": "paired_measurement_integrity_only",
            "complete": complete,
            "failures": failures,
            "passed": complete and not failures,
        },
        "complete": complete,
    }


def _paired_effects(
    reports: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    effects: dict[str, Any] = {}
    for name, before, after in _EFFECTS:
        if before not in reports or after not in reports:
            continue
        effects[name] = _paired_effect(
            name,
            _records(reports[before]),
            _records(reports[after]),
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    return effects


def _category_effects(
    reports: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    if not reports:
        return {}
    category_sets = [
        {
            str(record["evaluation"]["category"])
            for record in _records(report)
            if isinstance(record.get("evaluation"), dict)
        }
        for report in reports.values()
    ]
    categories = sorted(set.intersection(*category_sets)) if category_sets else []
    output: dict[str, Any] = {}
    for category in categories:
        selected = {
            variant: [
                record
                for record in _records(report)
                if record.get("evaluation", {}).get("category") == category
            ]
            for variant, report in reports.items()
        }
        effects: dict[str, Any] = {}
        for name, before, after in _EFFECTS:
            if before not in selected or after not in selected:
                continue
            effects[name] = _paired_effect(
                f"{category}:{name}",
                selected[before],
                selected[after],
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        output[category] = {
            "queries": len(next(iter(selected.values()))),
            "effects": effects,
        }
    return output


def _paired_effect(
    name: str,
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    before = {str(record["query_id"]): record for record in before_records}
    after = {str(record["query_id"]): record for record in after_records}
    if list(before) != list(after) or not before:
        raise ValueError(f"Paired query IDs differ or are empty for {name}.")
    query_ids = list(before)
    first = before[query_ids[0]]
    cutoffs = sorted(
        int(metric.rsplit("@", 1)[1])
        for metric in first["metrics"]["recall"]
        if metric.startswith("Recall@")
    )
    ranking_cutoff = max((value for value in cutoffs if value <= 10), default=min(cutoffs))
    recall_cutoff = max(cutoffs)
    dimensions = {
        "recall_at_max": ("recall", f"Recall@{recall_cutoff}"),
        "recall_ndcg": ("recall", f"NDCG@{ranking_cutoff}"),
        "final_recall_at_max": ("final", f"Recall@{recall_cutoff}"),
        "final_ndcg": ("final", f"NDCG@{ranking_cutoff}"),
        "delivered_recall_at_max": ("delivered", f"Recall@{recall_cutoff}"),
        "delivered_ndcg": ("delivered", f"NDCG@{ranking_cutoff}"),
    }
    quality = {
        dimension: paired_delta(
            [float(before[query_id]["metrics"][stage][metric]) for query_id in query_ids],
            [float(after[query_id]["metrics"][stage][metric]) for query_id in query_ids],
            bootstrap_samples=bootstrap_samples,
            seed=derived_seed(seed, name, dimension),
        )
        for dimension, (stage, metric) in dimensions.items()
    }
    latency = paired_delta(
        [float(before[query_id]["latency_seconds"]["fitz_sage"]) for query_id in query_ids],
        [float(after[query_id]["latency_seconds"]["fitz_sage"]) for query_id in query_ids],
        bootstrap_samples=bootstrap_samples,
        seed=derived_seed(seed, name, "latency"),
    )
    effect_name = name.rsplit(":", 1)[-1]
    return {
        "label": _EFFECT_LABELS[effect_name],
        "before": next(item[1] for item in _EFFECTS if item[0] == effect_name),
        "after": next(item[2] for item in _EFFECTS if item[0] == effect_name),
        "quality": quality,
        "latency_seconds": latency,
    }


def _measurement_failures(
    reports: dict[str, dict[str, Any]],
    requested_variants: list[str],
) -> list[str]:
    failures: list[str] = []
    reference: dict[str, Any] | None = None
    reference_variant = ""
    for variant, report in reports.items():
        if report.get("run", {}).get("ablation") != get_ablation(variant).as_dict():
            failures.append(f"{variant}: ablation metadata mismatch")
        if not report.get("complete") or not report.get("gate", {}).get("passed"):
            failures.append(f"{variant}: source report is incomplete or failed")
        result = report.get("dataset", {})
        identity = {
            "git": report.get("run", {}).get("git"),
            "selection": report.get("run", {}).get("selection"),
            "split_manifest": report.get("run", {}).get("split_manifest"),
            "dataset": result.get("dataset"),
            "collection": result.get("collection"),
            "index_mode": result.get("index_mode"),
            "cutoffs": result.get("selection", {}).get("cutoffs"),
            "query_ids": [record.get("query_id") for record in result.get("records", [])],
            "evaluation": [record.get("evaluation") for record in result.get("records", [])],
            "baseline": result.get("summary", {}).get("metrics", {}).get("baseline"),
        }
        if reference is None:
            reference = identity
            reference_variant = variant
        elif identity != reference:
            failures.append(f"{variant}: measurement identity differs from {reference_variant}")
    missing = [variant for variant in requested_variants if variant not in reports]
    failures.extend(f"{variant}: not completed" for variant in missing)
    return failures


def _records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = report.get("dataset", {}).get("records", [])
    if not isinstance(records, list):
        raise TypeError("Variant records must be a list.")
    return records


def _shared_value(reports: dict[str, dict[str, Any]], getter: Any) -> Any:
    values = [getter(report) for report in reports.values()]
    if not values:
        return None
    return values[0] if all(value == values[0] for value in values) else {"mixed": values}


def _variant_command(
    args: argparse.Namespace,
    *,
    variant: str,
    output: Path,
    markdown: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.fitz_bench.enterprise_rag_benchmark",
        "--ablation",
        variant,
        "--cache-dir",
        str(args.cache_dir),
        "--workspace-root",
        str(args.workspace_root),
        "--baseline-db",
        str(args.baseline_db),
        "--split-manifest",
        str(args.split_manifest),
        "--selection",
        args.selection,
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--index-mode",
        args.index_mode,
        "--reuse-workspace",
        "--reuse-index",
        "--max-download-gib",
        str(args.max_download_gib),
        "--max-extracted-gib",
        str(args.max_extracted_gib),
    ]
    command.append("--resume-queries" if args.resume_queries else "--no-resume-queries")
    if args.offline:
        command.append("--offline")
    if args.governance is not None:
        command.extend(["--governance", args.governance])
    for cutoff in args.cutoffs:
        command.extend(["--cutoff", str(cutoff)])
    return command


def _write_json(path: Path, report: dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    output = Path(path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(report), encoding="utf-8", newline="\n")


def _markdown(report: dict[str, Any]) -> str:
    cutoffs = [int(value) for value in report.get("cutoffs", [])]
    ranking_cutoff = max((value for value in cutoffs if value <= 10), default=min(cutoffs))
    recall_cutoff = max(cutoffs)
    lines = [
        "# EnterpriseRAG-Bench Component Ablation",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Selection: `{report['run']['selection']}`",
        f"- Gate: `{'PASS' if report['gate']['passed'] else 'FAIL'}`",
        "- Quality gate: `none`",
        "",
        (
            f"| Variant | Recall Recall@{recall_cutoff} | Final nDCG@{ranking_cutoff} | "
            f"Delivered nDCG@{ranking_cutoff} | Mean latency |"
        ),
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, variant in report["variants"].items():
        metrics = variant["metrics"]
        latency = variant["latency"]
        lines.append(
            f"| {name} | {metrics['recall'][f'Recall@{recall_cutoff}']:.4f} | "
            f"{metrics['final'][f'NDCG@{ranking_cutoff}']:.4f} | "
            f"{metrics['delivered'][f'NDCG@{ranking_cutoff}']:.4f} | "
            f"{latency['mean_seconds']:.2f}s |"
        )
    lines.extend(
        [
            "",
            "| Added component | Recall delta | Final nDCG delta | Latency delta |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for effect in report["effects"].values():
        lines.append(
            f"| {effect['label']} | "
            f"{_format_effect(effect['quality']['recall_at_max'])} | "
            f"{_format_effect(effect['quality']['final_ndcg'])} | "
            f"{_format_effect(effect['latency_seconds'], suffix='s')} |"
        )
    lines.extend(
        [
            "",
            "Component effects are descriptive. Broad recall is intentional, and no "
            "quality delta is used as a pass/fail threshold.",
            "",
        ]
    )
    return "\n".join(lines)


def _format_effect(effect: dict[str, Any], *, suffix: str = "") -> str:
    return (
        f"{effect['mean_delta']:+.4f}{suffix} "
        f"[{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}]"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", dest="variants", action="append", choices=ablation_names())
    parser.add_argument("--selection", choices=ALL_SELECTIONS, default="development")
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
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=Path("benchmarks/fixtures/enterprise_rag_split_v1.json"),
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/results/enterprise_rag_ablation_runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/enterprise_rag_ablation_latest.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmarks/results/enterprise_rag_ablation_latest.md"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--index-mode", choices=("source", "complete"), default="source")
    parser.add_argument("--governance")
    parser.add_argument(
        "--resume-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        type=int,
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parsed = parser.parse_args(argv)
    parsed.cutoffs = sorted(set(parsed.cutoffs or [1, 3, 5, 10, 20, 50]))
    if parsed.variants and len(set(parsed.variants)) != len(parsed.variants):
        parser.error("--variant values must be unique")
    if any(value < 1 for value in parsed.cutoffs):
        parser.error("--cutoff must be positive")
    if parsed.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

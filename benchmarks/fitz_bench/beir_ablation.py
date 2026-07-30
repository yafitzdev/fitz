"""Run paired BEIR ablations for Qwen query expansion and reranking."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.beir import DATASETS
from benchmarks.fitz_bench.beir_holdout import (
    load_query_manifest,
    query_manifest_digest,
)
from benchmarks.fitz_bench.retrieval_ablation import (
    ablation_names,
    get_ablation,
)

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


@dataclass(frozen=True)
class MetricDimension:
    name: str
    stage: str
    metric: str


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    query_manifest = (
        load_query_manifest(args.query_manifest) if args.query_manifest is not None else None
    )
    started = time.perf_counter()
    root = Path(__file__).resolve().parents[2]
    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    run_dir = (args.runs_dir / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    variants = args.variants or list(ablation_names())
    reports: dict[str, dict[str, Any]] = {}
    report_paths: dict[str, Path] = {}

    for index, variant in enumerate(variants, start=1):
        print(f"[{index}/{len(variants)}] Running BEIR ablation {variant}", flush=True)
        output = run_dir / f"{variant}.json"
        markdown = run_dir / f"{variant}.md"
        command = _variant_command(
            args,
            variant=variant,
            output=output,
            markdown=markdown,
        )
        completed = subprocess.run(command, cwd=root, check=False)
        if completed.returncode not in {0, 1} or not output.is_file():
            raise RuntimeError(
                f"BEIR ablation {variant!r} failed with exit code "
                f"{completed.returncode}; expected report {output}."
            )
        report = json.loads(output.read_text(encoding="utf-8"))
        if not isinstance(report, dict) or not report.get("complete"):
            raise RuntimeError(f"BEIR ablation {variant!r} did not produce a complete report.")
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
            query_manifest=query_manifest,
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
        query_manifest=query_manifest,
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
    query_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate independent variant reports and paired per-query effects."""
    failures = _measurement_failures(
        reports,
        requested_variants,
        query_manifest=query_manifest,
    )
    dataset_names = _shared_dataset_names(reports)
    datasets = [
        _dataset_summary(
            dataset,
            reports,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
        for dataset in dataset_names
    ]
    complete = set(reports) == set(requested_variants)
    return {
        "schema_version": 1,
        "run": {
            "run_id": run_id,
            "git": _shared_git_state(reports),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "requested_variants": requested_variants,
            "completed_variants": list(reports),
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": seed,
            "duration_seconds": time.perf_counter() - started,
            "query_manifest": (
                {
                    "name": query_manifest["name"],
                    "sha256": query_manifest_digest(query_manifest),
                }
                if query_manifest is not None
                else None
            ),
        },
        "method": {
            "paired_by": "dataset query_id",
            "confidence_interval": "deterministic paired percentile bootstrap, 95%",
            "index_reuse": "all variants query the same reusable Fitz-Sage collection",
            "process_isolation": (
                "each variant execution launches a fresh Python process; "
                "checkpoints may combine records across resumed executions"
            ),
            "constant_components": [
                "document projection and source index",
                "deterministic query planning",
                "typed retrieval and fusion",
                "candidate and read budgets",
                "evidence closure and compilation",
                "Pyrrho planning and decision",
            ],
        },
        "variants": {
            name: {
                **get_ablation(name).as_dict(),
                "report": str(report_paths[name].resolve()),
                "source_gate": reports[name].get("gate", {}),
            }
            for name in reports
        },
        "datasets": datasets,
        "macro": _macro_summary(datasets),
        "gate": {
            "type": "paired_measurement_integrity",
            "complete": complete,
            "failures": failures,
            "passed": complete and not failures,
        },
        "complete": complete,
    }


def _dataset_summary(
    dataset: str,
    reports: dict[str, dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    results = {variant: _dataset_result(report, dataset) for variant, report in reports.items()}
    first = next(iter(results.values()))
    dimensions = _metric_dimensions(first)
    variants = {
        variant: {
            "queries": int(result["selection"]["queries"]),
            "baseline": result["summary"]["metrics"]["baseline"],
            "metrics": result["summary"]["metrics"],
            "latency": result["fitz_sage"]["latency"],
            "timing": result["summary"].get("timing", {}),
        }
        for variant, result in results.items()
    }
    effects: dict[str, Any] = {}
    for effect_name, before_name, after_name in _EFFECTS:
        if before_name not in results or after_name not in results:
            continue
        effects[effect_name] = _paired_effect(
            dataset,
            effect_name,
            before_name,
            after_name,
            results[before_name]["records"],
            results[after_name]["records"],
            dimensions=dimensions,
            bootstrap_samples=bootstrap_samples,
            seed=seed,
        )
    subgroups = _subgroup_summaries(
        dataset,
        results,
        dimensions=dimensions,
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "name": dataset,
        "cutoffs": first["selection"]["cutoffs"],
        "variants": variants,
        "effects": effects,
        "subgroups": subgroups,
    }


def _paired_effect(
    dataset: str,
    effect_name: str,
    before_name: str,
    after_name: str,
    before_records: list[dict[str, Any]],
    after_records: list[dict[str, Any]],
    *,
    dimensions: list[MetricDimension],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    before = {record["query_id"]: record for record in before_records}
    after = {record["query_id"]: record for record in after_records}
    if set(before) != set(after):
        raise ValueError(f"{dataset} query IDs differ between {before_name} and {after_name}.")
    query_ids = list(before)
    quality = {
        dimension.name: paired_delta(
            [
                float(before[query_id]["metrics"][dimension.stage][dimension.metric])
                for query_id in query_ids
            ],
            [
                float(after[query_id]["metrics"][dimension.stage][dimension.metric])
                for query_id in query_ids
            ],
            bootstrap_samples=bootstrap_samples,
            seed=_derived_seed(seed, dataset, effect_name, dimension.name),
        )
        for dimension in dimensions
    }
    latency = paired_delta(
        [float(before[query_id]["latency_seconds"]["fitz_sage"]) for query_id in query_ids],
        [float(after[query_id]["latency_seconds"]["fitz_sage"]) for query_id in query_ids],
        bootstrap_samples=bootstrap_samples,
        seed=_derived_seed(seed, dataset, effect_name, "latency"),
    )
    return {
        "label": _EFFECT_LABELS[effect_name],
        "before": before_name,
        "after": after_name,
        "quality": quality,
        "latency_seconds": latency,
    }


def paired_delta(
    before: list[float],
    after: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    """Return a deterministic paired mean difference and percentile interval."""
    if len(before) != len(after) or not before:
        raise ValueError("Paired samples must have the same positive length.")
    deltas = [right - left for left, right in zip(before, after, strict=True)]
    observed = sum(deltas) / len(deltas)
    generator = random.Random(seed)
    bootstrap = sorted(
        sum(deltas[generator.randrange(len(deltas))] for _ in deltas) / len(deltas)
        for _ in range(bootstrap_samples)
    )
    low = _percentile(bootstrap, 0.025)
    high = _percentile(bootstrap, 0.975)
    direction = "inconclusive"
    if low > 0.0:
        direction = "positive"
    elif high < 0.0:
        direction = "negative"
    return {
        "observations": len(deltas),
        "before_mean": sum(before) / len(before),
        "after_mean": sum(after) / len(after),
        "mean_delta": observed,
        "ci95_low": low,
        "ci95_high": high,
        "direction": direction,
    }


def _subgroup_summaries(
    dataset: str,
    results: dict[str, dict[str, Any]],
    *,
    dimensions: list[MetricDimension],
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    group_sets = [
        {
            str(record["holdout"]["group"])
            for record in result["records"]
            if isinstance(record.get("holdout"), dict) and record["holdout"].get("group")
        }
        for result in results.values()
    ]
    shared_groups = set.intersection(*group_sets) if group_sets else set()
    groups = sorted(
        shared_groups,
        key=lambda group: (
            ("low", "medium", "high").index(group) if group in {"low", "medium", "high"} else 3,
            group,
        ),
    )
    output: dict[str, Any] = {}
    for group in groups:
        records = {
            variant: [
                record
                for record in result["records"]
                if record.get("holdout", {}).get("group") == group
            ]
            for variant, result in results.items()
        }
        first_records = next(iter(records.values()))
        effects: dict[str, Any] = {}
        for effect_name, before_name, after_name in _EFFECTS:
            if before_name not in records or after_name not in records:
                continue
            effects[effect_name] = _paired_effect(
                f"{dataset}/{group}",
                effect_name,
                before_name,
                after_name,
                records[before_name],
                records[after_name],
                dimensions=dimensions,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        output[group] = {
            "queries": len(first_records),
            "effects": effects,
        }
    return output


def _metric_dimensions(result: dict[str, Any]) -> list[MetricDimension]:
    cutoffs = [int(value) for value in result["selection"]["cutoffs"]]
    ranking_cutoff = max(
        (value for value in cutoffs if value <= 10),
        default=min(cutoffs),
    )
    recall_cutoff = max(cutoffs)
    return [
        MetricDimension("recall_ndcg", "recall", f"NDCG@{ranking_cutoff}"),
        MetricDimension("recall_at_max", "recall", f"Recall@{recall_cutoff}"),
        MetricDimension("final_ndcg", "final", f"NDCG@{ranking_cutoff}"),
        MetricDimension("delivered_ndcg", "delivered", f"NDCG@{ranking_cutoff}"),
    ]


def _measurement_failures(
    reports: dict[str, dict[str, Any]],
    requested_variants: list[str],
    *,
    query_manifest: dict[str, Any] | None = None,
) -> list[str]:
    failures: list[str] = []
    expected_manifest_sha256 = (
        query_manifest_digest(query_manifest) if query_manifest is not None else None
    )
    for variant, report in reports.items():
        expected = get_ablation(variant).as_dict()
        if report.get("run", {}).get("ablation") != expected:
            failures.append(f"{variant}: report ablation metadata mismatch")
        child_manifest = report.get("run", {}).get("query_manifest")
        child_manifest_sha256 = (
            child_manifest.get("sha256") if isinstance(child_manifest, dict) else None
        )
        if child_manifest_sha256 != expected_manifest_sha256:
            failures.append(f"{variant}: query manifest metadata mismatch")
        if not report.get("complete"):
            failures.append(f"{variant}: incomplete source report")
    if not reports:
        return failures

    git_states = {
        json.dumps(report.get("run", {}).get("git"), sort_keys=True) for report in reports.values()
    }
    if len(git_states) > 1:
        failures.append("variant reports were produced from different Git states")

    dataset_sets = {
        variant: tuple(
            result.get("dataset", {}).get("name") for result in report.get("datasets", [])
        )
        for variant, report in reports.items()
    }
    reference_variant = next(iter(dataset_sets))
    reference_datasets = dataset_sets[reference_variant]
    for variant, datasets in dataset_sets.items():
        if datasets != reference_datasets:
            failures.append(
                f"{variant}: datasets differ from {reference_variant}: "
                f"{datasets!r} != {reference_datasets!r}"
            )

    for dataset in reference_datasets:
        if not isinstance(dataset, str):
            continue
        reference_ids: list[str] | None = None
        reference_baseline: dict[str, float] | None = None
        reference_identity: tuple[Any, ...] | None = None
        reference_holdout: dict[str, Any] | None = None
        reference_variant = ""
        for variant, report in reports.items():
            try:
                result = _dataset_result(report, dataset)
            except KeyError:
                continue
            query_ids = [str(record["query_id"]) for record in result["records"]]
            baseline = result["summary"]["metrics"]["baseline"]
            identity = (
                result["dataset"].get("md5"),
                result.get("collection"),
                result.get("index_mode"),
                tuple(result["selection"].get("cutoffs", [])),
                result["selection"].get("query_limit"),
                json.dumps(result["selection"].get("query_manifest"), sort_keys=True),
            )
            holdout = {
                str(record["query_id"]): record.get("holdout") for record in result["records"]
            }
            if reference_ids is None:
                reference_ids = query_ids
                reference_baseline = baseline
                reference_identity = identity
                reference_holdout = holdout
                reference_variant = variant
                continue
            if query_ids != reference_ids:
                failures.append(
                    f"{dataset}/{variant}: query order differs from {reference_variant}"
                )
            if baseline != reference_baseline:
                failures.append(f"{dataset}/{variant}: plain BM25 differs from {reference_variant}")
            if identity != reference_identity:
                failures.append(
                    f"{dataset}/{variant}: dataset/index selection differs from {reference_variant}"
                )
            if holdout != reference_holdout:
                failures.append(
                    f"{dataset}/{variant}: holdout metadata differs from {reference_variant}"
                )

    missing = [variant for variant in requested_variants if variant not in reports]
    failures.extend(f"{variant}: not completed" for variant in missing)
    return failures


def _shared_dataset_names(reports: dict[str, dict[str, Any]]) -> list[str]:
    if not reports:
        return []
    ordered = [
        str(result["dataset"]["name"])
        for result in next(iter(reports.values())).get("datasets", [])
    ]
    available = [
        {str(result["dataset"]["name"]) for result in report.get("datasets", [])}
        for report in reports.values()
    ]
    return [dataset for dataset in ordered if all(dataset in names for names in available)]


def _shared_git_state(reports: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    states = [report.get("run", {}).get("git") for report in reports.values()]
    if not states:
        return None
    first = states[0]
    return first if all(state == first for state in states) else {"mixed": states}


def _dataset_result(report: dict[str, Any], dataset: str) -> dict[str, Any]:
    for result in report.get("datasets", []):
        if not isinstance(result, dict):
            raise TypeError("Ablation report dataset entries must be objects.")
        if result.get("dataset", {}).get("name") == dataset:
            return result
    raise KeyError(f"Dataset {dataset!r} is missing from a variant report.")


def _macro_summary(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    if not datasets:
        return {}
    variant_names = list(datasets[0]["variants"])
    variants: dict[str, Any] = {}
    for variant in variant_names:
        entries = [
            dataset["variants"][variant] for dataset in datasets if variant in dataset["variants"]
        ]
        variants[variant] = {
            "recall_ndcg": _mean_metric(entries, "recall", "NDCG"),
            "final_ndcg": _mean_metric(entries, "final", "NDCG"),
            "delivered_ndcg": _mean_metric(entries, "delivered", "NDCG"),
            "mean_latency_seconds": sum(
                float(entry["latency"]["mean_seconds"]) for entry in entries
            )
            / len(entries),
        }
    effects: dict[str, Any] = {}
    for effect_name, _before, _after in _EFFECTS:
        entries = [
            dataset["effects"][effect_name]
            for dataset in datasets
            if effect_name in dataset["effects"]
        ]
        if not entries:
            continue
        effects[effect_name] = {
            "recall_ndcg_delta": _mean_effect(entries, "recall_ndcg"),
            "final_ndcg_delta": _mean_effect(entries, "final_ndcg"),
            "delivered_ndcg_delta": _mean_effect(entries, "delivered_ndcg"),
            "mean_latency_delta_seconds": sum(
                float(entry["latency_seconds"]["mean_delta"]) for entry in entries
            )
            / len(entries),
        }
    return {
        "aggregation": "unweighted arithmetic mean across datasets",
        "variants": variants,
        "effects": effects,
    }


def _mean_metric(
    entries: list[dict[str, Any]],
    stage: str,
    metric_prefix: str,
) -> float:
    values = []
    for entry in entries:
        candidates = [
            (int(metric.rsplit("@", 1)[1]), float(value))
            for metric, value in entry["metrics"][stage].items()
            if metric.startswith(metric_prefix)
        ]
        preferred = [candidate for candidate in candidates if candidate[0] <= 10]
        values.append((max(preferred) if preferred else min(candidates))[1])
    return sum(values) / len(values)


def _mean_effect(entries: list[dict[str, Any]], dimension: str) -> float:
    return sum(float(entry["quality"][dimension]["mean_delta"]) for entry in entries) / len(entries)


def _derived_seed(base: int, *parts: str) -> int:
    payload = ":".join([str(base), *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("Cannot compute a percentile of an empty sequence.")
    index = round((len(values) - 1) * fraction)
    return values[index]


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
        "benchmarks.fitz_bench.beir_benchmark",
        "--ablation",
        variant,
        "--cache-dir",
        str(args.cache_dir),
        "--workspace-root",
        str(args.workspace_root),
        "--output",
        str(output),
        "--markdown",
        str(markdown),
        "--index-mode",
        args.index_mode,
        "--reuse-workspace",
        "--max-download-gib",
        str(args.max_download_gib),
        "--max-extracted-gib",
        str(args.max_extracted_gib),
    ]
    if args.resume_queries:
        command.append("--resume-queries")
    if args.offline:
        command.append("--offline")
    if args.query_limit is not None:
        command.extend(["--query-limit", str(args.query_limit)])
    if args.query_manifest is not None:
        command.extend(["--query-manifest", str(args.query_manifest)])
    if args.governance is not None:
        command.extend(["--governance", args.governance])
    for dataset in args.datasets or []:
        command.extend(["--dataset", dataset])
    for cutoff in args.cutoffs:
        command.extend(["--cutoff", str(cutoff)])
    return command


def _write_json(path: Path, report: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    output = path.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_markdown(report), encoding="utf-8")


def _markdown(report: dict[str, Any]) -> str:
    lines = [
        "# BEIR Component Ablation",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Gate: `{'PASS' if report['gate']['passed'] else 'FAIL'}`",
        (
            "- Confidence intervals: paired 95% percentile bootstrap "
            f"({report['run']['bootstrap_samples']} samples)"
        ),
        "",
        (
            "The local plain-BM25 row searches whole projected documents. Ablation "
            "variants retain Fitz-Sage's typed retrieval, deterministic query planning, "
            "evidence compilation, and Pyrrho."
        ),
        "",
    ]
    for dataset in report["datasets"]:
        lines.extend(_dataset_markdown(dataset))
    if report.get("macro"):
        lines.extend(_macro_markdown(report["macro"]))
    return "\n".join(lines).rstrip() + "\n"


def _dataset_markdown(dataset: dict[str, Any]) -> list[str]:
    cutoffs = [int(value) for value in dataset["cutoffs"]]
    ranking_cutoff = max((value for value in cutoffs if value <= 10), default=min(cutoffs))
    recall_cutoff = max(cutoffs)
    first = next(iter(dataset["variants"].values()))
    baseline = first["baseline"]
    lines = [
        f"## {dataset['name']}",
        "",
        (
            f"Plain BM25: nDCG@{ranking_cutoff} "
            f"`{baseline[f'NDCG@{ranking_cutoff}']:.4f}`, Recall@{recall_cutoff} "
            f"`{baseline[f'Recall@{recall_cutoff}']:.4f}`."
        ),
        "",
        (
            f"| Variant | Recall nDCG@{ranking_cutoff} | Final nDCG@{ranking_cutoff} | "
            f"Delivered nDCG@{ranking_cutoff} | Recall@{recall_cutoff} | "
            "Mean latency | p95 | Qwen mean | Rerank mean |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, variant in dataset["variants"].items():
        metrics = variant["metrics"]
        timing = variant.get("timing", {}).get("stage_groups", {})
        lines.append(
            "| {name} | {recall:.4f} | {final:.4f} | {delivered:.4f} | "
            "{recall_max:.4f} | {latency:.2f}s | {p95:.2f}s | {qwen:.2f}s | "
            "{rerank:.2f}s |".format(
                name=name,
                recall=metrics["recall"][f"NDCG@{ranking_cutoff}"],
                final=metrics["final"][f"NDCG@{ranking_cutoff}"],
                delivered=metrics["delivered"][f"NDCG@{ranking_cutoff}"],
                recall_max=metrics["recall"][f"Recall@{recall_cutoff}"],
                latency=variant["latency"]["mean_seconds"],
                p95=variant["latency"]["p95_seconds"],
                qwen=timing.get("semantic_expansion", {}).get("mean_seconds", 0.0),
                rerank=timing.get("rerank", {}).get("mean_seconds", 0.0),
            )
        )
    lines.extend(
        [
            "",
            (
                f"| Added component | Recall nDCG@{ranking_cutoff} delta | "
                f"Final nDCG@{ranking_cutoff} delta | "
                f"Delivered nDCG@{ranking_cutoff} delta | Mean latency delta |"
            ),
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for effect in dataset["effects"].values():
        lines.append(
            "| {label} | {recall} | {final} | {delivered} | {latency} |".format(
                label=effect["label"],
                recall=_format_effect(effect["quality"]["recall_ndcg"]),
                final=_format_effect(effect["quality"]["final_ndcg"]),
                delivered=_format_effect(effect["quality"]["delivered_ndcg"]),
                latency=_format_effect(effect["latency_seconds"], suffix="s"),
            )
        )
    lines.append("")
    if dataset.get("subgroups"):
        lines.extend(
            [
                "### Frozen Lexical-Overlap Strata",
                "",
                (
                    f"| Stratum | Queries | Qwen recall nDCG@{ranking_cutoff}, "
                    "no reranker | Qwen final nDCG, no reranker | "
                    "Qwen final nDCG, reranker on | Added latency, reranker on |"
                ),
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for group, subgroup in dataset["subgroups"].items():
            without = subgroup["effects"].get("expansion_without_reranker")
            with_reranker = subgroup["effects"].get("expansion_with_reranker")
            if without is None or with_reranker is None:
                continue
            lines.append(
                "| {group} | {queries} | {recall} | {final_without} | "
                "{final_with} | {latency} |".format(
                    group=group,
                    queries=subgroup["queries"],
                    recall=_format_effect(without["quality"]["recall_ndcg"]),
                    final_without=_format_effect(without["quality"]["final_ndcg"]),
                    final_with=_format_effect(with_reranker["quality"]["final_ndcg"]),
                    latency=_format_effect(
                        with_reranker["latency_seconds"],
                        suffix="s",
                    ),
                )
            )
        lines.append("")
    return lines


def _macro_markdown(macro: dict[str, Any]) -> list[str]:
    lines = [
        "## Macro Average",
        "",
        "| Variant | Recall nDCG | Final nDCG | Delivered nDCG | Mean latency |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, values in macro["variants"].items():
        lines.append(
            "| {name} | {recall:.4f} | {final:.4f} | {delivered:.4f} | {latency:.2f}s |".format(
                name=name,
                recall=values["recall_ndcg"],
                final=values["final_ndcg"],
                delivered=values["delivered_ndcg"],
                latency=values["mean_latency_seconds"],
            )
        )
    lines.extend(
        [
            "",
            "| Effect | Recall delta | Final delta | Delivered delta | Latency delta |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for name, values in macro["effects"].items():
        lines.append(
            "| {label} | {recall:+.4f} | {final:+.4f} | {delivered:+.4f} | "
            "{latency:+.2f}s |".format(
                label=_EFFECT_LABELS[name],
                recall=values["recall_ndcg_delta"],
                final=values["final_ndcg_delta"],
                delivered=values["delivered_ndcg_delta"],
                latency=values["mean_latency_delta_seconds"],
            )
        )
    lines.append("")
    return lines


def _format_effect(effect: dict[str, Any], *, suffix: str = "") -> str:
    return (
        f"{effect['mean_delta']:+.4f}{suffix} "
        f"[{effect['ci95_low']:+.4f}, {effect['ci95_high']:+.4f}]"
    )


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        dest="variants",
        action="append",
        choices=ablation_names(),
        help="Variant to run. Repeat; defaults to all four.",
    )
    parser.add_argument(
        "--dataset",
        dest="datasets",
        action="append",
        choices=sorted(DATASETS),
        help="Dataset to evaluate. Repeat; defaults to the standard three-dataset suite.",
    )
    parser.add_argument("--cache-dir", type=Path, default=Path(".benchmark-data/beir"))
    parser.add_argument("--workspace-root", type=Path, default=Path(".bench_workspace/beir"))
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("benchmarks/results/beir_ablation_runs"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/beir_ablation_latest.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("benchmarks/results/beir_ablation_latest.md"),
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--query-limit", type=int)
    parser.add_argument(
        "--query-manifest",
        type=Path,
        help="Frozen query selection manifest; cannot be combined with --query-limit.",
    )
    parser.add_argument("--index-mode", choices=("source", "complete"), default="source")
    parser.add_argument("--governance")
    parser.add_argument(
        "--resume-queries",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume variant-specific checkpoints; disable to replace them.",
    )
    parser.add_argument(
        "--cutoff",
        dest="cutoffs",
        action="append",
        type=int,
        help="Metric cutoff. Repeat; defaults to 1, 3, 5, 10, 20, and 50.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--max-download-gib", type=float, default=2.0)
    parser.add_argument("--max-extracted-gib", type=float, default=4.0)
    parsed = parser.parse_args(argv)
    parsed.cutoffs = sorted(set(parsed.cutoffs or [1, 3, 5, 10, 20, 50]))
    if parsed.query_limit is not None and parsed.query_limit < 1:
        parser.error("--query-limit must be positive")
    if parsed.query_manifest is not None and parsed.query_limit is not None:
        parser.error("--query-manifest cannot be combined with --query-limit")
    if any(value < 1 for value in parsed.cutoffs):
        parser.error("--cutoff must be positive")
    if parsed.bootstrap_samples < 100:
        parser.error("--bootstrap-samples must be at least 100")
    if parsed.max_download_gib <= 0 or parsed.max_extracted_gib <= 0:
        parser.error("download and extraction budgets must be positive")
    if parsed.variants and len(set(parsed.variants)) != len(parsed.variants):
        parser.error("--variant values must be unique")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())

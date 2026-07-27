# benchmarks/fitz_bench/runner.py
"""Command-line runner for the fitz-sage retrieval benchmark."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.fitz_bench.diagnostics import (
    compact_run,
    diagnose_case,
    evidence_signature,
)
from benchmarks.fitz_bench.distractors import stage_corpus
from benchmarks.fitz_bench.models import BenchmarkCase
from benchmarks.fitz_bench.validators import validate_case
from fitz_sage.config.loader import load_engine_config
from fitz_sage.core import Query
from fitz_sage.core.paths import FitzPaths
from fitz_sage.runtime import create_engine
from fitz_sage.storage.sqlite import SqliteConnectionManager


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval benchmark."""
    args = _parse_args(argv)
    root = Path(args.repo_root).resolve()
    source_corpus = (root / args.corpus).resolve()
    cases_path = (root / args.cases).resolve()
    output = (root / args.output).resolve()
    collection = args.collection or f"bench_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    workspace = _benchmark_workspace(root, args.workspace, collection)
    _activate_benchmark_workspace(workspace)
    corpus = source_corpus
    if args.distractors:
        corpus = stage_corpus(
            source_corpus,
            workspace / "staged_corpus",
            distractors=args.distractors,
        )

    cases = _load_cases(cases_path)
    cases = _select_cases(cases, args.case_ids)
    if args.limit is not None:
        cases = cases[: args.limit]

    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()

    engine: Any = _create_engine(args.engine, governance=args.governance)
    engine.load(collection)
    records: list[dict[str, Any]] = []
    manifest = None
    indexing_status: dict[str, Any] = {}
    ingestion_duration = 0.0
    try:
        ingestion_started = time.perf_counter()
        manifest = engine.point(
            corpus,
            collection=collection,
            start_worker=args.index_mode == "progressive",
        )
        if args.index_mode == "complete":
            engine.continue_indexing()
        indexing_status = dict(engine.indexing_status())
        ingestion_duration = time.perf_counter() - ingestion_started

        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            run = engine.trace(Query(text=case.query))
            pack = run.evidence
            pack_dict = pack.to_dict()
            validation = validate_case(
                case,
                pack_dict,
                ranked_items=_ranked_items(run),
                signals={"query": run.query.to_dict()},
            )
            duration = time.perf_counter() - case_started
            record = {
                "case": asdict(case),
                "duration_seconds": duration,
                "validation": validation.to_dict(),
                "diagnosis": diagnose_case(case, validation, run),
            }
            if args.report_detail == "full":
                record["evidence_pack"] = pack_dict
                record["retrieval_run"] = run.to_dict(include_content=True)
                record["signals"] = {
                    "query": run.query.to_dict(),
                    "query_profile": pack_dict.get("metadata", {}).get("query_profile", {}),
                    "retrieval_trace": pack_dict.get("metadata", {}).get("retrieval_trace", {}),
                    "evidence_compiler": pack_dict.get("metadata", {}).get("evidence_compiler", {}),
                    "evidence_delivery": pack_dict.get("metadata", {}).get("evidence_delivery", {}),
                    "pyrrho": pack_dict.get("metadata", {}).get("pyrrho", {}),
                }
            else:
                record["result"] = compact_run(run)
            records.append(record)
            status = "PASS" if validation.passed else "FAIL"
            print(
                f"[{index}/{len(cases)}] {status} {case.case_id} ({duration:.2f}s)",
                flush=True,
            )

        if args.reload_check:
            engine.stop_background_indexing()
            engine = None
            gc.collect()
            reloaded = _create_engine(args.engine, governance=args.governance)
            reloaded.load(collection)
            try:
                for index, (case, record) in enumerate(zip(cases, records), start=1):
                    original_signature = _record_signature(record)
                    replay_run = reloaded.trace(Query(text=case.query))
                    replay_validation = validate_case(
                        case,
                        replay_run.evidence.to_dict(),
                        ranked_items=_ranked_items(replay_run),
                        signals={"query": replay_run.query.to_dict()},
                    )
                    replay_signature = evidence_signature(replay_run)
                    record["stability"] = {
                        "retrieval_stable": (
                            original_signature["retrieval"] == replay_signature["retrieval"]
                        ),
                        "delivery_stable": (
                            original_signature["delivery"] == replay_signature["delivery"]
                        ),
                        "governance_stable": (
                            original_signature["mode"] == replay_signature["mode"]
                        ),
                        "validation": replay_validation.to_dict(),
                        "result": compact_run(replay_run),
                    }
                    print(
                        f"[reload {index}/{len(cases)}] {case.case_id}",
                        flush=True,
                    )
            finally:
                reloaded.stop_background_indexing()
    finally:
        if engine is not None:
            engine.stop_background_indexing()

    ingestion = _ingestion_report(manifest, indexing_status, ingestion_duration)
    summary = _summary(records)
    gate = _gate_result(
        summary,
        ingestion,
        metric=args.gate,
        minimum=args.minimum_pass_rate,
        allow_ingestion_failures=args.allow_ingestion_failures,
    )
    git_state = _git_state(root)
    report = {
        "run": {
            "run_id": run_id,
            **git_state,
            "collection": collection,
            "source_corpus": str(source_corpus),
            "corpus": str(corpus),
            "cases": str(cases_path),
            "index_mode": args.index_mode,
            "distractors": args.distractors,
            "reload_check": args.reload_check,
            "report_detail": args.report_detail,
            "governance_override": args.governance,
            "workspace": str(workspace),
            "duration_seconds": time.perf_counter() - started,
        },
        "ingestion": ingestion,
        "gate": gate,
        "summary": summary,
        "records": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        markdown_path = (root / args.markdown).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    print(json.dumps({"ingestion": ingestion["summary"], "gate": gate}, indent=2))
    return 0 if gate["passed"] else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run fitz-sage retrieval benchmarks.")
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument("--corpus", default="benchmarks/corpora/core", help="Corpus path.")
    parser.add_argument("--cases", default="benchmarks/cases/core.yaml", help="YAML cases.")
    parser.add_argument("--output", default="benchmarks/results/latest.json", help="JSON report.")
    parser.add_argument(
        "--markdown",
        default="benchmarks/results/latest.md",
        help="Optional Markdown summary path. Use empty string to skip.",
    )
    parser.add_argument("--collection", default=None, help="Collection name. Defaults unique.")
    parser.add_argument(
        "--workspace",
        default=None,
        help="Optional FITZ workspace path. Defaults to .bench_workspace/<collection>.",
    )
    parser.add_argument("--engine", default=None, help="Engine name passed to create_engine.")
    parser.add_argument(
        "--case-id",
        dest="case_ids",
        action="append",
        default=[],
        help="Run only this case id. Repeat to select multiple cases.",
    )
    parser.add_argument(
        "--governance",
        default=None,
        help=(
            "Optional governance provider override, e.g. 'pyrrho/C:\\path\\to\\pyrrho-v2-nano-g1'."
        ),
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit number of cases.")
    parser.add_argument(
        "--index-mode",
        choices=("complete", "progressive"),
        default="complete",
        help="Complete indexing before queries, or query while the worker runs.",
    )
    parser.add_argument(
        "--report-detail",
        choices=("compact", "full"),
        default="compact",
        help="Compact stage diagnostics, or full content-bearing retrieval runs.",
    )
    parser.add_argument(
        "--distractors",
        type=int,
        default=0,
        help="Add this many deterministic near-neighbor documents to a staged corpus.",
    )
    parser.add_argument(
        "--reload-check",
        action="store_true",
        help="Repeat every query after loading the persisted collection in a fresh engine.",
    )
    parser.add_argument(
        "--gate",
        choices=("full", "capability", "retrieval", "delivery", "query_shape", "none"),
        default="full",
        help="Metric used for the process exit code.",
    )
    parser.add_argument(
        "--minimum-pass-rate",
        type=float,
        default=1.0,
        help="Minimum selected metric rate required by the gate.",
    )
    parser.add_argument(
        "--allow-ingestion-failures",
        action="store_true",
        help="Measure supported-file indexing failures without failing the gate.",
    )
    parsed = parser.parse_args(argv)
    if parsed.distractors < 0:
        parser.error("--distractors must be non-negative")
    if not 0.0 <= parsed.minimum_pass_rate <= 1.0:
        parser.error("--minimum-pass-rate must be between 0 and 1")
    return parsed


def _benchmark_workspace(root: Path, workspace: str | None, collection: str) -> Path:
    """Return the isolated Fitz workspace for one benchmark run."""
    if workspace:
        path = Path(workspace)
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    return (root / ".bench_workspace" / collection).resolve()


def _activate_benchmark_workspace(workspace: Path) -> None:
    """Bind storage to this suite instead of retaining a prior singleton path."""
    SqliteConnectionManager.reset_instance()
    FitzPaths.set_workspace(workspace)


def _load_cases(path: Path) -> list[BenchmarkCase]:
    """Load benchmark cases from YAML."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    if not isinstance(raw, list):
        raise ValueError(f"Benchmark cases must be a YAML list: {path}")
    return [BenchmarkCase.from_dict(item) for item in raw]


def _select_cases(
    cases: list[BenchmarkCase],
    case_ids: list[str],
) -> list[BenchmarkCase]:
    """Select requested case ids while preserving suite order."""
    if not case_ids:
        return cases
    requested = set(case_ids)
    selected = [case for case in cases if case.case_id in requested]
    found = {case.case_id for case in selected}
    missing = requested - found
    if missing:
        raise ValueError(f"Unknown benchmark case id(s): {', '.join(sorted(missing))}")
    return selected


def _create_engine(engine: str | None, *, governance: str | None) -> Any:
    """Create a benchmark engine, optionally overriding the governance provider."""
    if governance is None:
        return create_engine(engine)

    engine_name = engine or "fitz_krag"
    config = load_engine_config(engine_name)
    values = config.model_dump()
    if "governance" not in values:
        raise ValueError(f"Engine '{engine_name}' does not expose a governance config field.")
    values["governance"] = governance
    return create_engine(engine_name, config=type(config)(**values))


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate benchmark summary."""
    summary = _record_summary(records)
    summary["by_domain"] = _group_summary(records, _record_domains)
    summary["by_tag"] = _group_summary(records, _record_tags)
    attribution: dict[str, int] = {}
    for record in records:
        for stage, count in record.get("diagnosis", {}).get("by_stage", {}).items():
            attribution[str(stage)] = attribution.get(str(stage), 0) + int(count)
    summary["failure_attribution"] = dict(sorted(attribution.items()))
    return summary


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for a record subset."""
    passed = sum(1 for record in records if record["validation"]["passed"])
    total = len(records)
    metrics = [record["validation"]["metrics"] for record in records]
    retrieval = [metric for metric in metrics if metric["retrieval_evaluated"]]
    delivery = [metric for metric in metrics if metric["delivery_evaluated"]]
    query_shape = [metric for metric in metrics if metric["query_shape_evaluated"]]
    capability = [metric for metric in metrics if metric["capability_evaluated"]]
    retrieval_passed = sum(1 for metric in retrieval if metric["retrieval_passed"])
    delivery_passed = sum(1 for metric in delivery if metric["delivery_passed"])
    query_shape_passed = sum(1 for metric in query_shape if metric["query_shape_passed"])
    capability_passed = sum(1 for metric in capability if metric["capability_passed"])
    governance = [metric for metric in metrics if metric["mode_match"] is not None]
    governance_passed = sum(1 for metric in governance if metric["mode_match"])
    stability = [record["stability"] for record in records if "stability" in record]
    retrieval_stable = sum(1 for item in stability if item["retrieval_stable"])
    delivery_stable = sum(1 for item in stability if item["delivery_stable"])
    governance_stable = sum(1 for item in stability if item["governance_stable"])
    ranking = [
        metric
        for record, metric in zip(records, metrics)
        if record["case"].get("required_evidence")
    ]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "retrieval_evaluated": len(retrieval),
        "retrieval_passed": retrieval_passed,
        "retrieval_failed": len(retrieval) - retrieval_passed,
        "retrieval_pass_rate": (retrieval_passed / len(retrieval) if retrieval else None),
        "delivery_evaluated": len(delivery),
        "delivery_passed": delivery_passed,
        "delivery_failed": len(delivery) - delivery_passed,
        "delivery_pass_rate": delivery_passed / len(delivery) if delivery else None,
        "query_shape_evaluated": len(query_shape),
        "query_shape_passed": query_shape_passed,
        "query_shape_failed": len(query_shape) - query_shape_passed,
        "query_shape_pass_rate": (query_shape_passed / len(query_shape) if query_shape else None),
        "capability_evaluated": len(capability),
        "capability_passed": capability_passed,
        "capability_failed": len(capability) - capability_passed,
        "capability_pass_rate": (capability_passed / len(capability) if capability else None),
        "governance_evaluated": len(governance),
        "governance_passed": governance_passed,
        "governance_failed": len(governance) - governance_passed,
        "governance_pass_rate": (governance_passed / len(governance) if governance else None),
        "stability_evaluated": len(stability),
        "retrieval_stability_rate": (retrieval_stable / len(stability) if stability else None),
        "delivery_stability_rate": (delivery_stable / len(stability) if stability else None),
        "governance_stability_rate": (governance_stable / len(stability) if stability else None),
        "ranking_evaluated": len(ranking),
        "mean_mrr": _optional_mean(metric["mrr"] for metric in ranking),
        "mean_required_recall": _optional_mean(metric["required_recall"] for metric in ranking),
        "hit_at_1_rate": _optional_mean(1.0 if metric["hit_at_1"] else 0.0 for metric in ranking),
        "hit_at_5_rate": _optional_mean(1.0 if metric["hit_at_5"] else 0.0 for metric in ranking),
        "forbidden_count": sum(int(metric["forbidden_count"]) for metric in retrieval),
    }


def _group_summary(records: list[dict[str, Any]], labels_for: Any) -> dict[str, Any]:
    """Build summaries grouped by a case label."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        for label in labels_for(record):
            grouped.setdefault(label, []).append(record)
    return {label: _record_summary(group) for label, group in sorted(grouped.items())}


def _record_domains(record: dict[str, Any]) -> tuple[str, ...]:
    """Return the domain label for one report record."""
    return (str(record["case"]["domain"]),)


def _record_tags(record: dict[str, Any]) -> tuple[str, ...]:
    """Return tag labels for one report record."""
    return tuple(str(tag) for tag in record["case"].get("tags", ()))


def _optional_mean(values: Any) -> float | None:
    """Compute mean for a generated numeric sequence."""
    collected = list(values)
    return sum(float(value) for value in collected) / len(collected) if collected else None


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary."""
    summary = report["summary"]
    lines = [
        "# fitz-sage Retrieval Benchmark",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Git: `{report['run']['git_sha']}`",
        f"- Cases: {summary['passed']}/{summary['total']} passed",
        (
            f"- Retrieval: {summary['retrieval_passed']}/"
            f"{summary['retrieval_evaluated']} "
            f"({_format_rate(summary['retrieval_pass_rate'])})"
        ),
        (
            f"- Governed delivery: {summary['delivery_passed']}/"
            f"{summary['delivery_evaluated']} "
            f"({_format_rate(summary['delivery_pass_rate'])})"
        ),
        (
            f"- Query shape: {summary['query_shape_passed']}/"
            f"{summary['query_shape_evaluated']} "
            f"({_format_rate(summary['query_shape_pass_rate'])})"
        ),
        (
            f"- Non-governance capability: {summary['capability_passed']}/"
            f"{summary['capability_evaluated']} "
            f"({_format_rate(summary['capability_pass_rate'])})"
        ),
        f"- Mean MRR: {_format_rate(summary['mean_mrr'])}",
        f"- Mean required recall: {_format_rate(summary['mean_required_recall'])}",
        f"- Ingestion healthy: {'yes' if report['ingestion']['summary']['healthy'] else 'no'}",
        f"- Gate: {'pass' if report['gate']['passed'] else 'fail'} ({report['gate']['metric']})",
        "",
        "## By Domain",
        "",
        "| Domain | Cases | Full | Retrieval | Delivery | Query shape | Capability |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for domain, domain_summary in summary["by_domain"].items():
        lines.append(
            (
                "| {domain} | {total} | {passed} | {retrieval} | "
                "{delivery} | {query_shape} | {capability} |"
            ).format(
                domain=domain,
                total=domain_summary["total"],
                passed=domain_summary["passed"],
                retrieval=_format_rate(domain_summary["retrieval_pass_rate"]),
                delivery=_format_rate(domain_summary["delivery_pass_rate"]),
                query_shape=_format_rate(domain_summary["query_shape_pass_rate"]),
                capability=_format_rate(domain_summary["capability_pass_rate"]),
            )
        )
    lines.extend(
        [
            "",
            "## Cases",
            "",
            "| Case | Domain | Passed | Mode | Failures |",
            "|---|---|---:|---|---|",
        ]
    )
    for record in report["records"]:
        case = record["case"]
        validation = record["validation"]
        failures = "<br>".join(validation["failures"])
        lines.append(
            "| {case_id} | {domain} | {passed} | {mode} | {failures} |".format(
                case_id=case["case_id"],
                domain=case["domain"],
                passed="yes" if validation["passed"] else "no",
                mode=_record_mode(record),
                failures=failures,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _record_mode(record: dict[str, Any]) -> str:
    result = record.get("result")
    if isinstance(result, dict):
        return str(result.get("mode") or "")
    pack = record.get("evidence_pack")
    if isinstance(pack, dict):
        return str(pack.get("mode") or "")
    return ""


def _ranked_items(run: Any) -> list[dict[str, Any]]:
    """Return content-bearing pre-governance evidence in validator form."""
    return [
        {
            "rank": item.rank,
            "file_path": item.file_path,
            "source_id": item.source_id,
            "address_kind": item.address_kind,
            "address_location": item.address_location,
            "excerpt": "",
            "content": item.content or "",
        }
        for item in run.ranked_evidence
    ]


def _record_signature(record: dict[str, Any]) -> dict[str, Any]:
    result = record.get("result")
    if isinstance(result, dict):
        retrieval_items = result.get("ranked_evidence")
        delivery_items = result.get("evidence")
    else:
        retrieval_run = record.get("retrieval_run")
        pack = record.get("evidence_pack")
        retrieval_items = (
            retrieval_run.get("ranked_evidence") if isinstance(retrieval_run, dict) else []
        )
        delivery_items = pack.get("items") if isinstance(pack, dict) else []
    return {
        "mode": _record_mode(record),
        "retrieval": _item_signature(retrieval_items),
        "delivery": _item_signature(delivery_items),
    }


def _item_signature(items: Any) -> list[tuple[str, str, str]]:
    items = items if isinstance(items, list) else []
    return [
        (
            str(item.get("file_path") or "").replace("\\", "/").lower(),
            str(item.get("address_kind") or ""),
            str(item.get("address_location") or ""),
        )
        for item in items
        if isinstance(item, dict)
    ]


def _ingestion_report(
    manifest: Any,
    status: dict[str, Any],
    duration_seconds: float = 0.0,
) -> dict[str, Any]:
    entries = manifest.entries() if manifest is not None else {}
    files = []
    by_extension: dict[str, dict[str, int]] = {}
    for entry in sorted(entries.values(), key=lambda item: item.rel_path):
        extension = entry.file_type or "<none>"
        extension_summary = by_extension.setdefault(extension, {"files": 0, "bytes": 0})
        extension_summary["files"] += 1
        extension_summary["bytes"] += int(entry.size_bytes)
        files.append(
            {
                "path": entry.rel_path,
                "extension": extension,
                "bytes": int(entry.size_bytes),
                "state": entry.state.value,
                "failure_stage": entry.failure_stage,
                "failure_message": entry.failure_message,
            }
        )
    summary = dict(status)
    summary["bytes"] = sum(item["bytes"] for item in files)
    summary["healthy"] = int(summary.get("failed", 0) or 0) == 0
    summary["duration_seconds"] = duration_seconds
    summary["files_per_second"] = len(files) / duration_seconds if duration_seconds > 0 else None
    summary["bytes_per_second"] = (
        summary["bytes"] / duration_seconds if duration_seconds > 0 else None
    )
    return {
        "summary": summary,
        "by_extension": dict(sorted(by_extension.items())),
        "files": files,
    }


def _gate_result(
    summary: dict[str, Any],
    ingestion: dict[str, Any],
    *,
    metric: str,
    minimum: float,
    allow_ingestion_failures: bool,
) -> dict[str, Any]:
    metric_keys = {
        "full": "pass_rate",
        "capability": "capability_pass_rate",
        "retrieval": "retrieval_pass_rate",
        "delivery": "delivery_pass_rate",
        "query_shape": "query_shape_pass_rate",
    }
    raw_actual = None if metric == "none" else summary[metric_keys[metric]]
    actual = float(raw_actual) if raw_actual is not None else None
    quality_passed = metric == "none" or (actual is not None and actual >= minimum)
    ingestion_passed = allow_ingestion_failures or bool(ingestion["summary"].get("healthy", False))
    return {
        "metric": metric,
        "minimum": minimum,
        "actual": actual,
        "quality_passed": quality_passed,
        "ingestion_passed": ingestion_passed,
        "passed": quality_passed and ingestion_passed,
    }


def _git_state(root: Path) -> dict[str, Any]:
    """Return commit, dirty state, and a content fingerprint for reproducibility."""
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        paths = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=root,
            stderr=subprocess.DEVNULL,
        ).split(b"\0")
        digest = hashlib.sha256()
        for raw_path in sorted(path for path in paths if path):
            relative = raw_path.decode("utf-8", errors="surrogateescape")
            digest.update(raw_path)
            digest.update(b"\0")
            try:
                digest.update((root / relative).read_bytes())
            except OSError as exc:
                digest.update(f"<unreadable:{type(exc).__name__}>".encode())
            digest.update(b"\0")
        return {
            "git_sha": git_sha,
            "git_dirty": bool(status.strip()),
            "worktree_sha256": digest.hexdigest(),
        }
    except Exception:
        return {
            "git_sha": "unknown",
            "git_dirty": None,
            "worktree_sha256": None,
        }


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    sys.exit(main())

# benchmarks/fitz_bench/runner.py
"""Command-line runner for the fitz-sage retrieval benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from benchmarks.fitz_bench.models import BenchmarkCase
from benchmarks.fitz_bench.validators import validate_case
from fitz_sage.config.loader import load_engine_config
from fitz_sage.core import Query
from fitz_sage.core.paths import FitzPaths
from fitz_sage.runtime import create_engine


def main(argv: list[str] | None = None) -> int:
    """Run the retrieval benchmark."""
    args = _parse_args(argv)
    root = Path(args.repo_root).resolve()
    corpus = (root / args.corpus).resolve()
    cases_path = (root / args.cases).resolve()
    output = (root / args.output).resolve()
    collection = args.collection or f"bench_{int(time.time())}_{uuid.uuid4().hex[:6]}"
    workspace = _benchmark_workspace(root, args.workspace, collection)
    FitzPaths.set_workspace(workspace)

    cases = _load_cases(cases_path)
    cases = _select_cases(cases, args.case_ids)
    if args.limit is not None:
        cases = cases[: args.limit]

    run_id = f"{int(time.time())}-{uuid.uuid4().hex[:8]}"
    started = time.perf_counter()

    engine = _create_engine(args.engine, governance=args.governance)
    engine.load(collection)
    records = []
    try:
        engine.point(corpus, collection=collection, start_worker=args.index_mode == "progressive")
        if args.index_mode == "complete":
            engine.continue_indexing()

        for index, case in enumerate(cases, start=1):
            case_started = time.perf_counter()
            pack = engine.evidence(Query(text=case.query))
            pack_dict = pack.to_dict()
            validation = validate_case(case, pack_dict)
            duration = time.perf_counter() - case_started
            records.append(
                {
                    "case": asdict(case),
                    "duration_seconds": duration,
                    "validation": validation.to_dict(),
                    "evidence_pack": pack_dict,
                    "signals": {
                        "query_profile": pack_dict.get("metadata", {}).get("query_profile", {}),
                        "retrieval_trace": pack_dict.get("metadata", {}).get("retrieval_trace", {}),
                        "evidence_compiler": pack_dict.get("metadata", {}).get(
                            "evidence_compiler", {}
                        ),
                        "governance_cutoff": pack_dict.get("metadata", {}).get(
                            "governance_cutoff", {}
                        ),
                    },
                }
            )
            status = "PASS" if validation.passed else "FAIL"
            print(
                f"[{index}/{len(cases)}] {status} {case.case_id} ({duration:.2f}s)",
                flush=True,
            )
    finally:
        engine.stop_background_indexing()

    report = {
        "run": {
            "run_id": run_id,
            "git_sha": _git_sha(root),
            "collection": collection,
            "corpus": str(corpus),
            "cases": str(cases_path),
            "index_mode": args.index_mode,
            "governance_override": args.governance,
            "workspace": str(workspace),
            "duration_seconds": time.perf_counter() - started,
        },
        "summary": _summary(records),
        "records": records,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown:
        markdown_path = (root / args.markdown).resolve()
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


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
    return parser.parse_args(argv)


def _benchmark_workspace(root: Path, workspace: str | None, collection: str) -> Path:
    """Return the isolated Fitz workspace for one benchmark run."""
    if workspace:
        path = Path(workspace)
        return path.resolve() if path.is_absolute() else (root / path).resolve()
    return (root / ".bench_workspace" / collection).resolve()


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
    return summary


def _record_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build aggregate metrics for a record subset."""
    passed = sum(1 for record in records if record["validation"]["passed"])
    total = len(records)
    metrics = [record["validation"]["metrics"] for record in records]
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": passed / total if total else 0.0,
        "mean_mrr": _mean(metric["mrr"] for metric in metrics),
        "mean_required_recall": _mean(metric["required_recall"] for metric in metrics),
        "hit_at_1_rate": _mean(1.0 if metric["hit_at_1"] else 0.0 for metric in metrics),
        "hit_at_5_rate": _mean(1.0 if metric["hit_at_5"] else 0.0 for metric in metrics),
        "forbidden_count": sum(int(metric["forbidden_count"]) for metric in metrics),
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


def _mean(values: Any) -> float:
    """Compute mean for a generated numeric sequence."""
    collected = list(values)
    return sum(float(value) for value in collected) / len(collected) if collected else 0.0


def _markdown(report: dict[str, Any]) -> str:
    """Render a compact Markdown summary."""
    summary = report["summary"]
    lines = [
        "# fitz-sage Retrieval Benchmark",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Git: `{report['run']['git_sha']}`",
        f"- Cases: {summary['passed']}/{summary['total']} passed",
        f"- Mean MRR: {summary['mean_mrr']:.3f}",
        f"- Mean required recall: {summary['mean_required_recall']:.3f}",
        "",
        "## By Domain",
        "",
        "| Domain | Cases | Passed | Pass Rate | Mean MRR | Mean Recall |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for domain, domain_summary in summary["by_domain"].items():
        lines.append(
            "| {domain} | {total} | {passed} | {pass_rate:.3f} | {mrr:.3f} | {recall:.3f} |".format(
                domain=domain,
                total=domain_summary["total"],
                passed=domain_summary["passed"],
                pass_rate=domain_summary["pass_rate"],
                mrr=domain_summary["mean_mrr"],
                recall=domain_summary["mean_required_recall"],
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
        pack = record["evidence_pack"]
        failures = "<br>".join(validation["failures"])
        lines.append(
            "| {case_id} | {domain} | {passed} | {mode} | {failures} |".format(
                case_id=case["case_id"],
                domain=case["domain"],
                passed="yes" if validation["passed"] else "no",
                mode=pack.get("mode"),
                failures=failures,
            )
        )
    lines.append("")
    return "\n".join(lines)


def _git_sha(root: Path) -> str:
    """Return current git SHA if available."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())

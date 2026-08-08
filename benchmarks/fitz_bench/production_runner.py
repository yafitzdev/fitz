"""Run the complete production-hardening benchmark matrix."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from benchmarks.fitz_bench import runner


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    root = Path(args.repo_root).resolve()
    suite_path = (root / args.suite).resolve()
    output_dir = (root / args.output_dir).resolve()
    aggregate_path = (root / args.output).resolve()
    markdown_path = (root / args.markdown).resolve()
    suites = _load_suites(suite_path)
    if args.suite_ids:
        suites = _select_suites(suites, args.suite_ids)

    started = time.perf_counter()
    reports: dict[str, dict[str, Any]] = {}
    for index, suite in enumerate(suites, start=1):
        suite_id = str(suite["id"])
        print(f"[suite {index}/{len(suites)}] {suite_id}", flush=True)
        report_path = output_dir / f"{suite_id}.json"
        report_markdown = output_dir / f"{suite_id}.md"
        runner_args = [
            "--repo-root",
            str(root),
            "--corpus",
            str(suite["corpus"]),
            "--cases",
            str(suite["cases"]),
            "--output",
            str(report_path),
            "--markdown",
            str(report_markdown),
            "--report-detail",
            "compact",
            "--gate",
            str(suite.get("gate", "retrieval")),
            "--minimum-pass-rate",
            str(suite.get("minimum_pass_rate", 1.0)),
        ]
        if args.governance:
            runner_args.extend(["--governance", args.governance])
        distractors = int(suite.get("distractors", 0) or 0)
        if distractors:
            runner_args.extend(["--distractors", str(distractors)])
        if suite.get("reload_check"):
            runner_args.append("--reload-check")
        if suite.get("allow_ingestion_failures"):
            runner_args.append("--allow-ingestion-failures")
        for case_id in suite.get("case_ids", []):
            runner_args.extend(["--case-id", str(case_id)])

        exit_code = runner.main(runner_args)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["matrix"] = {
            "required": bool(suite.get("required", False)),
            "runner_exit_code": exit_code,
            "compare_to": suite.get("compare_to"),
        }
        reports[suite_id] = report
        gc.collect()

    aggregate = _aggregate(
        suite_path=suite_path,
        reports=reports,
        duration=time.perf_counter() - started,
        governance=args.governance,
    )
    aggregate_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate_path.write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(_markdown(aggregate), encoding="utf-8")
    print(json.dumps(aggregate["summary"], indent=2))
    return 0 if aggregate["summary"]["production_gate_passed"] else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the production benchmark matrix.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--suite", default="benchmarks/suites/production.yaml")
    parser.add_argument("--output-dir", default="benchmarks/results/production")
    parser.add_argument("--output", default="benchmarks/results/production_latest.json")
    parser.add_argument("--markdown", default="benchmarks/results/production_latest.md")
    parser.add_argument("--governance", default=None)
    parser.add_argument(
        "--suite-id",
        dest="suite_ids",
        action="append",
        default=[],
        help="Run only this suite. Repeat to select multiple suites.",
    )
    return parser.parse_args(argv)


def _load_suites(path: Path) -> list[dict[str, Any]]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if raw.get("version") != 1 or not isinstance(raw.get("suites"), list):
        raise ValueError(f"Unsupported production suite manifest: {path}")
    suites = [dict(item) for item in raw["suites"] if isinstance(item, dict)]
    identifiers = [str(item.get("id") or "") for item in suites]
    if not all(identifiers) or len(set(identifiers)) != len(identifiers):
        raise ValueError("Production suite ids must be present and unique.")
    return suites


def _select_suites(
    suites: list[dict[str, Any]],
    suite_ids: list[str],
) -> list[dict[str, Any]]:
    requested = set(suite_ids)
    selected = [suite for suite in suites if str(suite["id"]) in requested]
    found = {str(suite["id"]) for suite in selected}
    missing = requested - found
    if missing:
        raise ValueError(f"Unknown production suite id(s): {', '.join(sorted(missing))}")
    return selected


def _aggregate(
    *,
    suite_path: Path,
    reports: dict[str, dict[str, Any]],
    duration: float,
    governance: str | None = None,
) -> dict[str, Any]:
    suite_summaries: dict[str, Any] = {}
    required_reports = [report for report in reports.values() if report["matrix"]["required"]]
    total = sum(int(report["summary"]["total"]) for report in required_reports)
    retrieval_evaluated = sum(
        int(report["summary"]["retrieval_evaluated"]) for report in required_reports
    )
    retrieval_passed = sum(
        int(report["summary"]["retrieval_passed"]) for report in required_reports
    )
    delivery_evaluated = sum(
        int(report["summary"]["delivery_evaluated"]) for report in required_reports
    )
    delivery_passed = sum(int(report["summary"]["delivery_passed"]) for report in required_reports)
    query_shape_evaluated = sum(
        int(report["summary"]["query_shape_evaluated"]) for report in required_reports
    )
    query_shape_passed = sum(
        int(report["summary"]["query_shape_passed"]) for report in required_reports
    )
    capability_evaluated = sum(
        int(report["summary"]["capability_evaluated"]) for report in required_reports
    )
    capability_passed = sum(
        int(report["summary"]["capability_passed"]) for report in required_reports
    )
    full_passed = sum(int(report["summary"]["passed"]) for report in required_reports)
    required_gate_passed = all(report["gate"]["passed"] for report in required_reports)

    regressions: list[dict[str, Any]] = []
    for suite_id, report in reports.items():
        regression = _regression(report, reports)
        if regression is not None:
            regressions.append(regression)
        suite_summaries[suite_id] = {
            "required": report["matrix"]["required"],
            "gate": report["gate"],
            "ingestion": report["ingestion"]["summary"],
            "summary": report["summary"],
            "regression": regression,
        }
    growth_regression_passed = all(
        not regression["retrieval_regressions"] and not regression["delivery_regressions"]
        for regression in regressions
    )

    return {
        "run": {
            "suite_manifest": str(suite_path),
            "duration_seconds": duration,
            "governance_model": _governance_identity(governance),
        },
        "summary": {
            "required_suites": len(required_reports),
            "measured_suites": len(reports),
            "required_cases": total,
            "retrieval_evaluated": retrieval_evaluated,
            "retrieval_passed": retrieval_passed,
            "retrieval_pass_rate": (
                retrieval_passed / retrieval_evaluated if retrieval_evaluated else None
            ),
            "delivery_evaluated": delivery_evaluated,
            "delivery_passed": delivery_passed,
            "delivery_pass_rate": (
                delivery_passed / delivery_evaluated if delivery_evaluated else None
            ),
            "query_shape_evaluated": query_shape_evaluated,
            "query_shape_passed": query_shape_passed,
            "query_shape_pass_rate": (
                query_shape_passed / query_shape_evaluated if query_shape_evaluated else None
            ),
            "capability_evaluated": capability_evaluated,
            "capability_passed": capability_passed,
            "capability_pass_rate": (
                capability_passed / capability_evaluated if capability_evaluated else None
            ),
            "full_passed": full_passed,
            "full_pass_rate": full_passed / total if total else 0.0,
            "suite_gates_passed": required_gate_passed,
            "growth_regression_passed": growth_regression_passed,
            "production_gate_passed": (required_gate_passed and growth_regression_passed),
        },
        "suites": suite_summaries,
    }


def _regression(
    report: dict[str, Any],
    reports: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    baseline_id = report["matrix"].get("compare_to")
    if not baseline_id or baseline_id not in reports:
        return None
    baseline = _case_metrics(reports[baseline_id])
    current = _case_metrics(report)
    shared = sorted(set(baseline) & set(current))
    regressed = [
        case_id
        for case_id in shared
        if baseline[case_id]["retrieval_passed"] and not current[case_id]["retrieval_passed"]
    ]
    improved = [
        case_id
        for case_id in shared
        if not baseline[case_id]["retrieval_passed"] and current[case_id]["retrieval_passed"]
    ]
    delivery_regressed = [
        case_id
        for case_id in shared
        if baseline[case_id]["delivery_passed"] and not current[case_id]["delivery_passed"]
    ]
    delivery_improved = [
        case_id
        for case_id in shared
        if not baseline[case_id]["delivery_passed"] and current[case_id]["delivery_passed"]
    ]
    return {
        "baseline": baseline_id,
        "shared_cases": len(shared),
        "retrieval_regressions": regressed,
        "retrieval_improvements": improved,
        "delivery_regressions": delivery_regressed,
        "delivery_improvements": delivery_improved,
    }


def _case_metrics(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(record["case"]["case_id"]): dict(record["validation"]["metrics"])
        for record in report["records"]
    }


def _governance_identity(spec: str | None) -> dict[str, Any]:
    identity: dict[str, Any] = {"spec": spec or "configured-default"}
    if not spec or not spec.startswith("pyrrho/"):
        return identity
    package = Path(spec.split("/", 1)[1]).expanduser()
    identity["package"] = str(package.resolve()) if package.exists() else str(package)
    identity["local_package"] = package.is_dir()
    if not package.is_dir():
        return identity

    manifest_path = package / "manifest.json"
    manifest = _read_json(manifest_path)
    release = manifest.get("release") if isinstance(manifest, dict) else None
    preferred = release.get("preferred_onnx_graph") if isinstance(release, dict) else None
    graph_path = package / str(preferred or "model.onnx")
    parity_path = package / "onnx_parity_report.json"
    if isinstance(manifest, dict):
        parity = manifest.get("onnx_parity")
        if isinstance(parity, dict) and isinstance(parity.get("report"), str):
            parity_path = package / parity["report"]
    parity_report = _read_json(parity_path)
    comparisons = parity_report.get("comparisons") if isinstance(parity_report, dict) else None
    identity.update(
        {
            "manifest": {
                "present": manifest_path.is_file(),
                "sha256": _sha256_file(manifest_path),
            },
            "selected_graph": {
                "path": str(graph_path),
                "present": graph_path.is_file(),
                "sha256": _sha256_file(graph_path),
            },
            "parity_report": {
                "path": str(parity_path),
                "present": parity_path.is_file(),
                "sha256": _sha256_file(parity_path),
                "passed": (
                    parity_report.get("passed") if isinstance(parity_report, dict) else None
                ),
                "comparisons": {
                    str(name): {
                        "passed": value.get("passed"),
                        "decision_differences": value.get("decision_differences"),
                        "max_probability_error": value.get("max_probability_error"),
                    }
                    for name, value in (
                        comparisons.items() if isinstance(comparisons, dict) else ()
                    )
                    if isinstance(value, dict)
                },
            },
        }
    )
    return identity


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# fitz-sage Production Hardening Matrix",
        "",
        f"- Required suites: {summary['required_suites']}",
        f"- Required cases: {summary['required_cases']}",
        f"- Retrieval pass rate: {_format_rate(summary['retrieval_pass_rate'])}",
        f"- Governed delivery pass rate: {_format_rate(summary['delivery_pass_rate'])}",
        f"- Query-shape pass rate: {_format_rate(summary['query_shape_pass_rate'])}",
        f"- Non-governance capability pass rate: {_format_rate(summary['capability_pass_rate'])}",
        f"- Full governed pass rate: {summary['full_pass_rate']:.3f}",
        "- Corpus-growth regressions: "
        + ("none" if summary["growth_regression_passed"] else "present"),
        ("- Production gate: " + ("PASS" if summary["production_gate_passed"] else "FAIL")),
        "",
        "## Suites",
        "",
        "| Suite | Required | Cases | Retrieval | Delivery | Query shape | Full | Gate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for suite_id, suite in report["suites"].items():
        metrics = suite["summary"]
        lines.append(
            (
                "| {suite_id} | {required} | {total} | {retrieval} | "
                "{delivery} | {query_shape} | {full:.3f} | {gate} |"
            ).format(
                suite_id=suite_id,
                required="yes" if suite["required"] else "no",
                total=metrics["total"],
                retrieval=_format_rate(metrics["retrieval_pass_rate"]),
                delivery=_format_rate(metrics["delivery_pass_rate"]),
                query_shape=_format_rate(metrics["query_shape_pass_rate"]),
                full=metrics["pass_rate"],
                gate="pass" if suite["gate"]["passed"] else "fail",
            )
        )
        regression = suite.get("regression")
        if regression and regression["retrieval_regressions"]:
            lines.append(
                f"\nRetrieval regressions against `{regression['baseline']}`: "
                + ", ".join(f"`{item}`" for item in regression["retrieval_regressions"])
                + "\n"
            )
        if regression and regression["delivery_regressions"]:
            lines.append(
                f"\nDelivery regressions against `{regression['baseline']}`: "
                + ", ".join(f"`{item}`" for item in regression["delivery_regressions"])
                + "\n"
            )
    lines.append("")
    return "\n".join(lines)


def _format_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


if __name__ == "__main__":
    sys.exit(main())

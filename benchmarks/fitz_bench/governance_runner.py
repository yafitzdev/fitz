"""Run Pyrrho over the balanced fixed-evidence governance benchmark."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.governance_cases import build_cases
from benchmarks.fitz_bench.models import normalize_mode
from fitz_sage.integrations.pyrrho import create_pyrrho


def main(argv: list[str] | None = None) -> int:
    """Run the balanced governance benchmark."""
    args = _parse_args(argv)
    root = Path(args.repo_root).resolve()
    output = (root / args.output).resolve()
    markdown = (root / args.markdown).resolve() if args.markdown else None
    cases = build_cases()
    if args.limit is not None:
        cases = cases[: args.limit]

    pyrrho = create_pyrrho(args.pyrrho)
    started = time.perf_counter()
    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        case_started = time.perf_counter()
        decision = pyrrho.decide(case.query, list(case.contexts))
        actual_mode = normalize_mode(decision.verdict)
        expected_mode = normalize_mode(case.expected_mode)
        passed = actual_mode == expected_mode
        records.append(
            {
                "case": case.to_dict(),
                "duration_seconds": time.perf_counter() - case_started,
                "prediction": decision.to_dict(),
                "expected_mode": expected_mode,
                "actual_mode": actual_mode,
                "passed": passed,
            }
        )
        if args.verbose:
            status = "PASS" if passed else "FAIL"
            print(
                f"{index:03d}/{len(cases):03d} {status} "
                f"expected={expected_mode} actual={actual_mode} {case.case_id}",
                flush=True,
            )

    report: dict[str, Any] = {
        "run": {
            "run_id": f"{int(time.time())}-{uuid.uuid4().hex[:8]}",
            "git_sha": _git_sha(root),
            "pyrrho_override": args.pyrrho,
            "duration_seconds": time.perf_counter() - started,
            "benchmark": "governance_balanced_fixed_evidence",
        },
        "summary": _summary(records),
        "records": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if markdown is not None:
        markdown.parent.mkdir(parents=True, exist_ok=True)
        markdown.write_text(_markdown(report), encoding="utf-8")

    print(json.dumps(report["summary"], indent=2))
    return 0 if report["summary"]["failed"] == 0 else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run balanced fixed-evidence Pyrrho governance benchmark."
    )
    parser.add_argument("--repo-root", default=".", help="Repository root.")
    parser.add_argument(
        "--pyrrho",
        default="pyrrho",
        help="Pyrrho provider spec, e.g. pyrrho/C:\\path\\to\\best_model.",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/results/governance_balanced_latest.json",
        help="JSON report path.",
    )
    parser.add_argument(
        "--markdown",
        default="benchmarks/results/governance_balanced_latest.md",
        help="Markdown report path. Use empty string to skip.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional case limit.")
    parser.add_argument("--verbose", action="store_true", help="Print per-case predictions.")
    return parser.parse_args(argv)


def _summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(records)
    passed = sum(1 for record in records if record["passed"])
    expected_counts = Counter(record["expected_mode"] for record in records)
    actual_counts = Counter(record["actual_mode"] for record in records)
    confusion = Counter((record["expected_mode"], record["actual_mode"]) for record in records)
    by_expected = {}
    recalls = []
    for label in sorted(expected_counts):
        label_records = [record for record in records if record["expected_mode"] == label]
        label_passed = sum(1 for record in label_records if record["passed"])
        recall = label_passed / len(label_records) if label_records else 0.0
        recalls.append(recall)
        by_expected[label] = {
            "total": len(label_records),
            "passed": label_passed,
            "recall": recall,
        }

    unsafe_expected = {"disputed", "insufficient"}
    unsafe_records = [record for record in records if record["expected_mode"] in unsafe_expected]
    false_sufficient = sum(1 for record in unsafe_records if record["actual_mode"] == "sufficient")
    sufficient_records = [record for record in records if record["expected_mode"] == "sufficient"]
    false_reject = sum(1 for record in sufficient_records if record["actual_mode"] != "sufficient")

    by_domain = defaultdict(list)
    for record in records:
        by_domain[record["case"]["domain"]].append(record)

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "accuracy": passed / total if total else 0.0,
        "macro_recall": sum(recalls) / len(recalls) if recalls else 0.0,
        "false_sufficient_count": false_sufficient,
        "false_sufficient_rate": false_sufficient / len(unsafe_records) if unsafe_records else 0.0,
        "false_reject_sufficient_count": false_reject,
        "false_reject_sufficient_rate": (
            false_reject / len(sufficient_records) if sufficient_records else 0.0
        ),
        "expected_counts": dict(sorted(expected_counts.items())),
        "actual_counts": dict(sorted(actual_counts.items())),
        "by_expected": by_expected,
        "by_domain": {
            domain: {
                "total": len(items),
                "passed": sum(1 for item in items if item["passed"]),
                "accuracy": (
                    sum(1 for item in items if item["passed"]) / len(items) if items else 0.0
                ),
            }
            for domain, items in sorted(by_domain.items())
        },
        "confusion": {
            f"{expected}->{actual}": count
            for (expected, actual), count in sorted(confusion.items())
        },
    }


def _markdown(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Balanced Governance Benchmark",
        "",
        f"- Run: `{report['run']['run_id']}`",
        f"- Pyrrho: `{report['run']['pyrrho_override']}`",
        f"- Cases: {summary['passed']}/{summary['total']} passed",
        f"- Accuracy: {summary['accuracy']:.3f}",
        f"- Macro recall: {summary['macro_recall']:.3f}",
        f"- False sufficient rate: {summary['false_sufficient_rate']:.3f}",
        f"- False reject sufficient rate: {summary['false_reject_sufficient_rate']:.3f}",
        "",
        "## By Expected Mode",
        "",
        "| Expected | Cases | Passed | Recall |",
        "|---|---:|---:|---:|",
    ]
    for label, item in summary["by_expected"].items():
        lines.append(f"| {label} | {item['total']} | {item['passed']} | {item['recall']:.3f} |")
    lines.extend(["", "## Confusion", "", "| Expected -> Actual | Count |", "|---|---:|"])
    for pair, count in summary["confusion"].items():
        lines.append(f"| {pair} | {count} |")
    lines.extend(["", "## Failed Cases", "", "| Case | Expected | Actual |", "|---|---|---|"])
    for record in report["records"]:
        if record["passed"]:
            continue
        lines.append(
            f"| {record['case']['case_id']} | {record['expected_mode']} | {record['actual_mode']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _git_sha(root: Path) -> str:
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

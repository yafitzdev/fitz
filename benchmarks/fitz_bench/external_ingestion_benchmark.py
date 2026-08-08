"""Run ingestion and crash recovery against official NapierOne files."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from benchmarks.fitz_bench.ingestion_benchmark import run_benchmark
from benchmarks.fitz_bench.napierone import DEFAULT_TYPES, prepare_corpus
from benchmarks.fitz_bench.recovery_benchmark import run_recovery_benchmark


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    metadata = prepare_corpus(
        args.cache_dir,
        file_types=args.file_types or DEFAULT_TYPES,
        profile=args.profile,
        variant=args.variant,
        max_download_bytes=int(args.max_download_gib * 1024**3),
        max_extracted_bytes=int(args.max_extracted_gib * 1024**3),
        offline=args.offline,
        progress=print,
    )
    corpus = Path(metadata.corpus_dir)

    if args.download_only:
        report = {"corpus": metadata.as_dict()}
        return _render(report, args.output, passed=True)

    with _workspace(args.workspace) as workspace:
        baseline_workspace = workspace / "baseline"
        baseline = run_benchmark(
            corpus,
            workspace=baseline_workspace,
            iterations=args.iterations,
            parser=args.parser,
            target_files_per_second=args.target_files_per_second,
            max_failure_rate=args.max_failure_rate,
        )
        expected = _expected_signature(baseline["iterations"][0])
        recovery = None
        if not args.skip_recovery:
            recovery = run_recovery_benchmark(
                corpus,
                workspace=workspace / "recovery",
                parser=args.parser,
                crash_after_indexed=args.crash_after_indexed,
                expected=expected,
            )

        passed = bool(baseline["gate"]["passed"]) and (recovery is None or bool(recovery["passed"]))
        report = {
            "corpus": metadata.as_dict(),
            "baseline": baseline,
            "recovery": recovery,
            "gate": {
                "baseline_passed": bool(baseline["gate"]["passed"]),
                "recovery_passed": recovery is None or bool(recovery["passed"]),
                "passed": passed,
            },
        }
        return _render(report, args.output, passed=passed)


def _expected_signature(run: dict[str, Any]) -> dict[str, Any]:
    return {
        "indexed_files": int(run["indexed_files"]),
        "failed_files": int(run["failed_files"]),
        "unsupported_files": int(run["unsupported_files"]),
        "indexed_bytes": int(run["indexed_bytes"]),
        "by_extension": run["by_extension"],
        "sqlite_counts": run["sqlite_counts"],
    }


class _workspace:
    def __init__(self, selected: Path | None) -> None:
        self._selected = selected
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        if self._selected is not None:
            workspace = self._selected.resolve()
            workspace.mkdir(parents=True, exist_ok=True)
            return workspace
        self._temporary = tempfile.TemporaryDirectory(prefix="fitz-external-ingestion-")
        return Path(self._temporary.name) / ".fitz"

    def __exit__(self, *_: object) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


def _render(report: dict[str, Any], output: Path | None, *, passed: bool) -> int:
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 1


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".benchmark-data/napierone"),
        help="Download/extraction cache outside the versioned benchmark corpus.",
    )
    parser.add_argument("--profile", choices=("tiny", "small", "total"), default="tiny")
    parser.add_argument(
        "--variant",
        choices=("standard", "nomagic", "password"),
        default="standard",
    )
    parser.add_argument(
        "--type",
        dest="file_types",
        action="append",
        help="NapierOne type to include. Repeat; defaults to supported document types.",
    )
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--max-download-gib", type=float, default=5.0)
    parser.add_argument("--max-extracted-gib", type=float, default=20.0)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument(
        "--parser",
        choices=("cpu", "docling", "docling_vision", "glm_ocr"),
        default="cpu",
    )
    parser.add_argument("--target-files-per-second", type=float, default=1.0)
    parser.add_argument(
        "--max-failure-rate",
        type=float,
        default=0.05,
        help="Maximum supported-file parser failure rate accepted for real corpus data.",
    )
    parser.add_argument("--skip-recovery", action="store_true")
    parser.add_argument("--crash-after-indexed", type=int, default=10)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(main())

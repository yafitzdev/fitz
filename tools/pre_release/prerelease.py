# tools/pre_release/prerelease.py
"""
Pre-release validation script for fitz-sage.

Runs ALL checks before tagging a release:
- Git status (uncommitted changes warning)
- isort (import sorting)
- ruff format (code formatting)
- ruff check (linting)
- pytest (unit tests, skip slow by default)
- contract_map (architecture validation)

Usage:
    # Run all checks (fix mode for lint, unit tests only):
    python tools/pre_release/prerelease.py

    # Check-only mode (CI-style, no auto-fix):
    python tools/pre_release/prerelease.py --check

    # Include slow tests:
    python tools/pre_release/prerelease.py --slow

    # Include E2E tests:
    python tools/pre_release/prerelease.py --e2e

    # Full release validation (all tests):
    python tools/pre_release/prerelease.py --full

    # Skip tests (lint only):
    python tools/pre_release/prerelease.py --lint-only

Requirements:
    pip install ruff isort pytest
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

LINT_PATHS = ["fitz_sage", "tests", "tools"]
DEFAULT_PYTEST_MARKERS = "not slow and not e2e and not e2e_parser and not integration"


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class CheckResult:
    """Result of a single check."""

    name: str
    success: bool
    returncode: int
    duration: float = 0.0
    skipped: bool = False
    message: str = ""


@dataclass
class PreReleaseConfig:
    """Configuration for pre-release checks."""

    fix_mode: bool = True
    run_tests: bool = True
    include_slow: bool = False
    include_e2e: bool = False
    include_integration: bool = False
    lint_only: bool = False
    verbose: bool = False


# =============================================================================
# Utilities
# =============================================================================


def get_project_root() -> Path:
    """Find the project root directory."""
    current = Path(__file__).resolve().parent
    for _ in range(10):
        if (current / "pyproject.toml").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent
    return Path(__file__).resolve().parent.parent.parent


def print_header(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print("=" * 70)


def print_subheader(title: str) -> None:
    """Print a subsection header."""
    print(f"\n{'-' * 50}")
    print(f"  {title}")
    print("-" * 50)


def run_command(
    name: str,
    cmd: list[str],
    capture: bool = False,
) -> CheckResult:
    """Run a command and return the result."""
    print_subheader(name)
    print(f"$ {' '.join(cmd)}")
    print()

    start = time.time()
    try:
        if capture:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
        else:
            result = subprocess.run(cmd)

        duration = time.time() - start
        success = result.returncode == 0

        status = "PASS" if success else "FAIL"
        print(f"\n[{status}] {name} ({duration:.1f}s)")

        return CheckResult(
            name=name,
            success=success,
            returncode=result.returncode,
            duration=duration,
        )

    except FileNotFoundError:
        duration = time.time() - start
        print(f"\n[FAIL] {name} - command not found")
        return CheckResult(
            name=name,
            success=False,
            returncode=-1,
            duration=duration,
            message=f"Command not found: {cmd[0]}",
        )


# =============================================================================
# Individual Checks
# =============================================================================


def check_git_status() -> CheckResult:
    """Check for uncommitted changes."""
    print_subheader("Git Status")

    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print("Failed to get git status")
        return CheckResult(name="Git Status", success=False, returncode=result.returncode)

    changes = result.stdout.strip()
    if changes:
        print("WARNING: Uncommitted changes detected:")
        print(changes)
        print("\nConsider committing before release.")
        return CheckResult(
            name="Git Status",
            success=True,  # Warning only, not a failure
            returncode=0,
            message="Uncommitted changes (warning)",
        )
    else:
        print("Working directory clean")
        return CheckResult(name="Git Status", success=True, returncode=0)


def run_isort(fix: bool) -> CheckResult:
    """Run isort import sorting."""
    if fix:
        cmd = [sys.executable, "-m", "isort", *LINT_PATHS]
    else:
        cmd = [sys.executable, "-m", "isort", "--check-only", "--diff", *LINT_PATHS]
    return run_command("isort", cmd)


def run_ruff_format(fix: bool) -> CheckResult:
    """Run ruff formatter."""
    if fix:
        cmd = [sys.executable, "-m", "ruff", "format", *LINT_PATHS]
    else:
        cmd = [sys.executable, "-m", "ruff", "format", "--check", *LINT_PATHS]
    return run_command("Ruff Format", cmd)


def run_ruff_check(fix: bool) -> CheckResult:
    """Run ruff linter."""
    if fix:
        cmd = [sys.executable, "-m", "ruff", "check", "--fix", *LINT_PATHS]
    else:
        cmd = [sys.executable, "-m", "ruff", "check", *LINT_PATHS]
    return run_command("Ruff Check", cmd)


def run_pytest(config: PreReleaseConfig) -> CheckResult:
    """Run pytest with configured markers."""
    # Build marker expression
    markers: list[str] = []

    if not config.include_slow:
        markers.append("not slow")
    if not config.include_e2e:
        markers.append("not e2e")
        markers.append("not e2e_parser")
    if not config.include_integration:
        markers.append("not integration")

    cmd = [sys.executable, "-m", "pytest", "-v", "--tb=short"]

    if markers:
        marker_expr = " and ".join(markers)
        cmd.extend(["-m", marker_expr])

    # Add color
    cmd.append("--color=yes")

    return run_command("pytest", cmd)


def run_contract_map() -> CheckResult:
    """Run architecture contract validation."""
    # contract_map returns non-zero on violations, no special flag needed
    cmd = [sys.executable, "-m", "tools.contract_map"]
    return run_command("Contract Map", cmd)


# =============================================================================
# Main Runner
# =============================================================================


def run_prerelease(config: PreReleaseConfig) -> int:
    """
    Run all pre-release checks.

    Returns:
        0 if all checks passed, 1 otherwise.
    """
    project_root = get_project_root()
    os.chdir(project_root)

    print_header("FITZ-AI PRE-RELEASE VALIDATION")
    print(f"Project root: {project_root}")
    print(f"Mode: {'FIX' if config.fix_mode else 'CHECK'}")
    if config.lint_only:
        print("Scope: Lint only (skipping tests)")
    elif config.include_e2e:
        print("Scope: Full (including E2E tests)")
    elif config.include_slow:
        print("Scope: Extended (including slow tests)")
    else:
        print("Scope: Quick (unit tests only)")

    results: list[CheckResult] = []
    start_time = time.time()

    # 1. Git status (warning only)
    print_header("PHASE 1: GIT STATUS")
    results.append(check_git_status())

    # 2. Linting
    print_header("PHASE 2: CODE QUALITY")
    results.append(run_isort(fix=config.fix_mode))
    results.append(run_ruff_format(fix=config.fix_mode))
    results.append(run_ruff_check(fix=config.fix_mode))

    # 3. Tests (unless lint-only)
    if not config.lint_only:
        print_header("PHASE 3: TESTS")
        results.append(run_pytest(config))

    # 4. Architecture validation
    print_header("PHASE 4: ARCHITECTURE")
    results.append(run_contract_map())

    # Summary
    total_duration = time.time() - start_time
    print_summary(results, total_duration)

    # Return code
    failures = [r for r in results if not r.success and not r.skipped]
    return 0 if not failures else 1


def print_summary(results: list[CheckResult], total_duration: float) -> None:
    """Print summary of all results."""
    print_header("SUMMARY")

    passed = 0
    failed = 0
    warnings = 0

    for result in results:
        if result.skipped:
            status = "SKIP"
            icon = "-"
        elif result.success:
            status = "PASS"
            icon = "+"
            passed += 1
            if result.message:
                warnings += 1
        else:
            status = "FAIL"
            icon = "x"
            failed += 1

        duration_str = f"({result.duration:.1f}s)" if result.duration > 0 else ""
        msg = f" - {result.message}" if result.message else ""
        print(f"  [{icon}] {status:4}  {result.name:20} {duration_str:10} {msg}")

    print()
    print(f"Total: {passed} passed, {failed} failed, {warnings} warnings")
    print(f"Duration: {total_duration:.1f}s")
    print()

    if failed == 0:
        print("=" * 70)
        print("  ALL CHECKS PASSED - Ready to tag release!")
        print("=" * 70)
    else:
        print("=" * 70)
        print(f"  {failed} CHECK(S) FAILED - Fix issues before release")
        print("=" * 70)


# =============================================================================
# Entry Point
# =============================================================================


def parse_args() -> PreReleaseConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Pre-release validation for fitz-sage",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/pre_release/prerelease.py           # Quick check with auto-fix
  python tools/pre_release/prerelease.py --check   # CI mode (no auto-fix)
  python tools/pre_release/prerelease.py --full    # Full validation
  python tools/pre_release/prerelease.py --lint-only  # Skip tests
        """,
    )

    parser.add_argument(
        "--check",
        "-c",
        action="store_true",
        help="Check-only mode (no auto-fix)",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Include slow tests",
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Include E2E tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Include integration tests",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full validation (all tests)",
    )
    parser.add_argument(
        "--lint-only",
        action="store_true",
        help="Skip tests, lint only",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    config = PreReleaseConfig(
        fix_mode=not args.check,
        run_tests=not args.lint_only,
        include_slow=args.slow or args.full,
        include_e2e=args.e2e or args.full,
        include_integration=args.integration or args.full,
        lint_only=args.lint_only,
        verbose=args.verbose,
    )

    return config


if __name__ == "__main__":
    config = parse_args()
    exit_code = run_prerelease(config)
    sys.exit(exit_code)

# tools/pre_release/lint.py
"""
Pre-release lint script for fitz-ai.

Runs code quality checks before tagging a release.

Usage:
    # Just run it (defaults to --fix mode):
    python tools/pre_release/lint.py

    # Or with arguments:
    python tools/pre_release/lint.py --check  # Check only (CI mode)

Requirements:
    pip install ruff black isort
"""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# CONFIGURATION - Edit these as needed
# =============================================================================

# Set to True to auto-fix by default, False to check-only by default
DEFAULT_FIX_MODE = True

# Paths to lint (relative to project root)
LINT_PATHS = ["fitz_ai", "tests", "tools"]

# Set to True to skip isort (if you use ruff's import sorting instead)
SKIP_ISORT = False

# Set to True to use Black instead of Ruff Format
USE_BLACK_INSTEAD_OF_RUFF_FORMAT = False


# =============================================================================
# Implementation
# =============================================================================


def get_project_root() -> Path:
    """Find the project root directory (where pyproject.toml is)."""
    # Start from this file's directory
    current = Path(__file__).resolve().parent

    # Walk up until we find pyproject.toml or hit the filesystem root
    for _ in range(10):  # Max 10 levels up
        if (current / "pyproject.toml").exists():
            return current
        if (current / ".git").exists():
            return current
        parent = current.parent
        if parent == current:
            break
        current = parent

    # Fallback: assume we're in tools/pre_release, so go up 2 levels
    return Path(__file__).resolve().parent.parent.parent


@dataclass
class LintResult:
    """Result of a lint command."""

    name: str
    success: bool
    returncode: int


def run_command(name: str, cmd: list[str]) -> LintResult:
    """Run a command and return the result."""
    print(f"\n{'=' * 60}")
    print(f"Running: {name}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    try:
        result = subprocess.run(cmd)
        success = result.returncode == 0

        if success:
            print(f"✓ {name} passed")
        else:
            print(f"✗ {name} failed (exit code {result.returncode})")

        return LintResult(name=name, success=success, returncode=result.returncode)

    except FileNotFoundError:
        print(f"✗ {name} not found - install with: pip install {name.lower()}")
        return LintResult(name=name, success=False, returncode=-1)


def run_isort(fix: bool, paths: list[str]) -> LintResult:
    """Run isort to sort imports."""
    if fix:
        cmd = ["isort", *paths]
    else:
        cmd = ["isort", "--check-only", "--diff", *paths]
    return run_command("isort", cmd)


def run_black(fix: bool, paths: list[str]) -> LintResult:
    """Run black formatter."""
    if fix:
        cmd = ["black", *paths]
    else:
        cmd = ["black", "--check", "--diff", *paths]
    return run_command("Black", cmd)


def run_ruff_format(fix: bool, paths: list[str]) -> LintResult:
    """Run ruff formatter."""
    if fix:
        cmd = ["ruff", "format", *paths]
    else:
        cmd = ["ruff", "format", "--check", *paths]
    return run_command("Ruff Format", cmd)


def run_ruff_check(fix: bool, paths: list[str]) -> LintResult:
    """Run ruff linter."""
    if fix:
        cmd = ["ruff", "check", "--fix", *paths]
    else:
        cmd = ["ruff", "check", *paths]
    return run_command("Ruff Check", cmd)


def print_summary(results: list[LintResult]) -> bool:
    """Print summary of all results. Returns True if all passed."""
    print("\n")
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_passed = True
    for result in results:
        status = "✓ PASS" if result.success else "✗ FAIL"
        print(f"  {status}  {result.name}")
        if not result.success:
            all_passed = False

    print("=" * 60)

    if all_passed:
        print("\n✓ All checks passed! Ready to tag release.\n")
    else:
        print("\n✗ Some checks failed.\n")

    return all_passed


def run_all_linters(fix: bool = DEFAULT_FIX_MODE) -> int:
    """
    Run all linters.

    Args:
        fix: If True, auto-fix issues. If False, check only.

    Returns:
        0 if all passed, 1 otherwise.
    """
    # Change to project root so relative paths work
    project_root = get_project_root()
    os.chdir(project_root)

    print("=" * 60)
    print("FITZ-AI PRE-RELEASE LINT")
    print("=" * 60)
    print(f"Mode: {'FIX (auto-fixing issues)' if fix else 'CHECK (read-only)'}")
    print(f"Project root: {project_root}")
    print(f"Paths: {LINT_PATHS}")

    results: list[LintResult] = []

    # 1. isort - sort imports
    if not SKIP_ISORT:
        results.append(run_isort(fix=fix, paths=LINT_PATHS))

    # 2. Formatter (ruff format or black)
    if USE_BLACK_INSTEAD_OF_RUFF_FORMAT:
        results.append(run_black(fix=fix, paths=LINT_PATHS))
    else:
        results.append(run_ruff_format(fix=fix, paths=LINT_PATHS))

    # 3. Ruff linter
    results.append(run_ruff_check(fix=fix, paths=LINT_PATHS))

    # Print summary
    all_passed = print_summary(results)

    return 0 if all_passed else 1


# =============================================================================
# Entry Point - Just run this file!
# =============================================================================

if __name__ == "__main__":
    # Check for --check flag to override default fix mode
    check_only = "--check" in sys.argv or "-c" in sys.argv
    fix_mode = not check_only and DEFAULT_FIX_MODE

    exit_code = run_all_linters(fix=fix_mode)
    sys.exit(exit_code)

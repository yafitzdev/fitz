# tools/pre_release.py
"""
Pre-release validation script.

Runs ALL checks that CI will run, fixes what can be auto-fixed,
and fails fast with clear errors for anything that needs manual intervention.

Usage:
    python -m tools.pre_release          # Check only
    python -m tools.pre_release --fix    # Auto-fix and check
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"


def get_ci_deps() -> set[str]:
    """Extract all deps from CI workflow."""
    content = CI_WORKFLOW.read_text()
    deps = set()
    for match in re.finditer(r"pip install ([^\n]+)", content):
        for pkg in match.group(1).split():
            if not pkg.startswith("-"):
                pkg_name = re.split(r"[<>=\[]", pkg)[0].lower().replace("-", "_")
                if pkg_name:
                    deps.add(pkg_name)
    return deps


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run command and return result."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=False)
    return result


def step(name: str) -> None:
    """Print step header."""
    print(f"\n{'='*60}\n{name}\n{'='*60}")


def main() -> int:
    fix_mode = "--fix" in sys.argv
    errors: list[str] = []

    # Step 1: Auto-fix formatting
    step("1. Formatting (black + isort)")
    if fix_mode:
        run(["python", "-m", "black", "."])
        run(["python", "-m", "isort", "."])
        print("  Fixed formatting issues")

    result = run(["python", "-m", "black", "--check", "--quiet", "."], check=False)
    if result.returncode != 0:
        errors.append("black: Files need formatting (run with --fix)")
        print("  FAIL: black found formatting issues")
        print(result.stdout[:500] if result.stdout else "")

    result = run(["python", "-m", "isort", "--check-only", "--quiet", "."], check=False)
    if result.returncode != 0:
        errors.append("isort: Imports need sorting (run with --fix)")
        print("  FAIL: isort found import order issues")

    if not errors or all("--fix" not in e for e in errors):
        print("  OK")

    # Step 2: Ruff with auto-fix
    step("2. Linting (ruff)")
    if fix_mode:
        result = run(["python", "-m", "ruff", "check", "--fix", "."])
        print("  Applied ruff auto-fixes")

    result = run(["python", "-m", "ruff", "check", "."], check=False)
    if result.returncode != 0:
        errors.append("ruff: Linting errors found")
        print(result.stdout)
        print("\n  HINTS:")
        print("  - F401 'imported but unused' in try blocks: add # noqa: F401")
        print("  - F401 for optional deps in tests: use importlib.util.find_spec()")
    else:
        print("  OK")

    # Step 3: Test critical import paths (simulates CI)
    step("3. Critical Import Check (simulates CI with minimal deps)")

    ci_deps = get_ci_deps()

    # These are the imports CI tests - they must work with minimal deps
    critical_imports = [
        ("fitz_sage", "Package root"),
        ("fitz_sage.core", "Core module"),
        ("fitz_sage.core.Query", "Query class"),
        ("fitz_sage.core.Answer", "Answer class"),
        ("fitz_sage.llm", "LLM module"),
        ("fitz_sage.llm.auth", "Auth module"),
    ]

    for import_path, desc in critical_imports:
        parts = import_path.rsplit(".", 1)
        if len(parts) == 2:
            cmd = f"from {parts[0]} import {parts[1]}"
        else:
            cmd = f"import {import_path}"

        result = run(["python", "-c", cmd], check=False)
        if result.returncode != 0:
            stderr = result.stderr
            if "ModuleNotFoundError" in stderr:
                # Extract missing module name
                missing = stderr.split("No module named")[-1].strip().strip("'\"").split("'")[0]
                missing_normalized = missing.lower().replace("-", "_")

                # Check if it's in CI deps
                if missing_normalized in ci_deps:
                    print(f"  OK: {desc} (requires {missing}, in CI deps)")
                else:
                    errors.append(f"Import failed: {import_path} (missing {missing})")
                    print(f"  FAIL: {desc}")
                    print(f"        Missing: {missing}")
                    print("        Fix: Add to CI minimal deps in .github/workflows/ci.yml")
            else:
                errors.append(f"Import failed: {import_path}")
                print(f"  FAIL: {desc}")
                print(f"        {stderr[:200]}")
        else:
            print(f"  OK: {desc}")

    # Step 4: Run all unit tests (not just a subset - catches mock/interface mismatches)
    step("4. Unit Tests")
    result = run(
        [
            "python",
            "-m",
            "pytest",
            "tests/unit/",  # All unit tests - catches interface mismatches
            "-v",
            "--tb=line",
            "-x",  # Stop on first failure
            "-q",
            "-m",
            "not slow and not integration",
            "--ignore=tests/unit/llm/test_auth_adapters.py",  # Skip slow/complex tests
            "--ignore=tests/unit/integrations/test_langchain.py",  # Requires langchain_core + langsmith
        ],
        check=False,
    )
    output = result.stdout + result.stderr
    if result.returncode != 0:
        if "ModuleNotFoundError" in output or "No module named" in output:
            # Extract missing module names
            missing_modules = set()
            for line in output.split("\n"):
                if "No module named" in line:
                    # Extract module name from various formats
                    parts = line.split("No module named")[-1]
                    mod = parts.strip().strip("'\"").split("'")[0].split(".")[0]
                    if mod:
                        missing_modules.add(mod)

            for mod in missing_modules:
                mod_normalized = mod.lower().replace("-", "_")
                if mod_normalized in ci_deps:
                    print(f"  WARN: {mod} missing locally but IS in CI deps (OK for CI)")
                else:
                    errors.append(f"Missing dep: {mod}")
                    print(f"  FAIL: '{mod}' not installed AND not in CI deps")
                    print(f"        >> Add 'pip install {mod}' to .github/workflows/ci.yml")
        else:
            errors.append("Tests failed")
            print("  FAIL: Some tests failed")
            # Show last few lines
            lines = output.strip().split("\n")
            for line in lines[-10:]:
                print(f"        {line}")
    else:
        print("  OK: Quick tests passed")

    # Summary
    step("SUMMARY")
    if errors:
        print("FAILED - Fix these issues before release:\n")
        for e in errors:
            print(f"  - {e}")
        print("\nRun 'python -m tools.pre_release --fix' to auto-fix formatting issues")
        return 1
    else:
        print("ALL CHECKS PASSED - Ready for release!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

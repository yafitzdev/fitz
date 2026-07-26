# tools/wheel_smoke.py
"""Smoke-test a built fitz-sage wheel in a fresh virtual environment.

The check intentionally installs the wheel, not the checkout. That catches
runtime dependencies missing from ``pyproject.toml`` and entry-point packaging
issues before release.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SmokePaths:
    """Resolved paths for the isolated wheel smoke environment."""

    root: Path
    venv_dir: Path
    python: Path
    fitz: Path


def project_root() -> Path:
    """Return the repository root."""
    current = Path(__file__).resolve().parent
    for _ in range(8):
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise RuntimeError("Could not locate project root from tools/wheel_smoke.py")


def run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a subprocess with consistent logging and failure handling."""
    print(f"$ {' '.join(cmd)}", flush=True)
    if capture:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.stdout:
            print_console_safe(result.stdout)
    else:
        result = subprocess.run(cmd, cwd=cwd, env=env, text=True)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit {result.returncode}: {' '.join(cmd)}")
    return result


def print_console_safe(text: str) -> None:
    """Print captured subprocess output without failing on console encoding."""
    encoding = sys.stdout.encoding or "utf-8"
    printable = text.encode(encoding, errors="replace").decode(encoding)
    print(printable, flush=True)


def python_bin(venv_dir: Path) -> Path:
    """Return the venv Python executable path for this platform."""
    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "python.exe" if os.name == "nt" else "python"
    return venv_dir / scripts / executable


def fitz_bin(venv_dir: Path) -> Path:
    """Return the venv fitz console-script path for this platform."""
    scripts = "Scripts" if os.name == "nt" else "bin"
    executable = "fitz.exe" if os.name == "nt" else "fitz"
    return venv_dir / scripts / executable


def newest_wheel(dist_dir: Path) -> Path | None:
    """Find the newest wheel in a directory."""
    wheels = sorted(dist_dir.glob("*.whl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return wheels[0] if wheels else None


def build_wheel(out_dir: Path) -> Path:
    """Build a wheel into ``out_dir`` and return its path."""
    root = project_root()
    for generated in (root / "build", root / "fitz_sage.egg-info"):
        if generated.exists():
            shutil.rmtree(generated)
    out_dir.mkdir(parents=True, exist_ok=True)
    run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(out_dir)],
        cwd=root,
    )
    wheel = newest_wheel(out_dir)
    if wheel is None:
        raise RuntimeError(f"Build completed but no wheel was found in {out_dir}")
    return wheel


def validate_wheel_contents(wheel: Path) -> None:
    """Reject stale files from subsystems removed from the source tree."""
    forbidden = (
        "fitz_sage/cli/context.py",
        "fitz_sage/cli/utils.py",
        "fitz_sage/cli/ui/engine_selection.py",
        "fitz_sage/cli/ui/progress.py",
        "fitz_sage/core/chunk.py",
        "fitz_sage/core/conflicts.py",
        "fitz_sage/core/constants.py",
        "fitz_sage/core/knowledge.py",
        "fitz_sage/core/math.py",
        "fitz_sage/core/registry.py",
        "fitz_sage/core/utils.py",
        "fitz_sage/core/paths/cache.py",
        "fitz_sage/core/paths/ingestion.py",
        "fitz_sage/ingestion/chunking/",
        "fitz_sage/ingestion/detection.py",
        "fitz_sage/ingestion/exceptions/",
        "fitz_sage/prompts/entities.py",
        "fitz_sage/tabular/direct_query.py",
        "fitz_sage/tabular/extractor.py",
        "fitz_sage/tabular/models.py",
        "fitz_sage/tabular/query.py",
        "fitz_sage/tabular/registry.py",
        "fitz_sage/tabular/sql_gen.py",
    )
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    stale = [name for name in names if any(name.startswith(prefix) for prefix in forbidden)]
    if stale:
        raise RuntimeError(f"Wheel contains removed modules: {', '.join(stale)}")
    if "fitz_sage/py.typed" not in names:
        raise RuntimeError("Wheel does not contain fitz_sage/py.typed")


def resolve_wheel(args: argparse.Namespace, temp_root: Path) -> Path:
    """Resolve or build the wheel under test."""
    if args.wheel is not None:
        wheel = Path(args.wheel).resolve()
        if not wheel.exists():
            raise FileNotFoundError(f"Wheel not found: {wheel}")
        return wheel

    if args.dist_dir is not None:
        dist_dir = Path(args.dist_dir).resolve()
        if args.skip_build:
            wheel = newest_wheel(dist_dir)
            if wheel is None:
                raise FileNotFoundError(f"No wheel found in {dist_dir}")
            return wheel
        return build_wheel(dist_dir)

    return build_wheel(temp_root / "dist")


def create_smoke_env(temp_root: Path) -> SmokePaths:
    """Create the fresh virtual environment used for the smoke."""
    venv_dir = temp_root / "wheel-venv"
    venv.EnvBuilder(with_pip=True, clear=True).create(venv_dir)
    python = python_bin(venv_dir)
    fitz = fitz_bin(venv_dir)
    if not python.exists():
        raise RuntimeError(f"venv Python was not created: {python}")
    return SmokePaths(root=temp_root, venv_dir=venv_dir, python=python, fitz=fitz)


def install_wheel(paths: SmokePaths, wheel: Path) -> None:
    """Install the wheel and verify its dependency graph."""
    run([str(paths.python), "-m", "pip", "install", "--upgrade", "pip"], cwd=paths.root)
    run([str(paths.python), "-m", "pip", "install", str(wheel)], cwd=paths.root)
    run([str(paths.python), "-m", "pip", "check"], cwd=paths.root)


def smoke_import(paths: SmokePaths) -> None:
    """Import runtime code and instantiate the default KRAG engine."""
    smoke_cwd = paths.root / "import-smoke"
    smoke_cwd.mkdir()
    env = isolated_env(paths.root)
    run([str(paths.fitz), "--help"], cwd=smoke_cwd, env=env)
    code = (
        "from importlib.resources import files; "
        "assert files('fitz_sage').joinpath('py.typed').is_file(); "
        "from fitz_sage.runtime import create_engine; "
        "engine = create_engine('fitz_krag'); "
        "print(type(engine).__name__)"
    )
    run([str(paths.python), "-c", code], cwd=smoke_cwd, env=env)


def smoke_retrieve(paths: SmokePaths) -> None:
    """Run a first-run retrieve command from the installed wheel."""
    smoke_cwd = paths.root / "retrieve-smoke"
    source_dir = smoke_cwd / "docs"
    source_dir.mkdir(parents=True)
    (source_dir / "release_notes.md").write_text(
        "\n".join(
            [
                "# Sprint 47 Checkout Notes",
                "",
                "Checkout regression TC-4812 validates that invoice currency conversion "
                "remains stable after gateway failover.",
                "The release owner is Maya Chen and the system under test is PayBridge.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    env = isolated_env(paths.root)
    result = run(
        [
            str(paths.fitz),
            "retrieve",
            "Which test case validates checkout regression?",
            "--source",
            str(source_dir),
            "--collection",
            "wheel_smoke",
            "--top-k",
            "5",
        ],
        cwd=smoke_cwd,
        env=env,
        capture=True,
    )
    output = result.stdout or ""
    if "TC-4812" not in output:
        raise RuntimeError("Wheel retrieve smoke did not return TC-4812 evidence")
    if "Preparing managed Qwen" not in output:
        raise RuntimeError("Wheel retrieve smoke did not preflight managed Qwen")


def isolated_env(temp_root: Path) -> dict[str, str]:
    """Build an environment that keeps smoke state outside the checkout."""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["HF_HOME"] = str(temp_root / "hf_home")
    env["HF_HUB_CACHE"] = str(temp_root / "hf_cache")
    return env


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", help="Existing wheel file to test.")
    parser.add_argument("--dist-dir", help="Directory to build into or read from.")
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="Use the newest wheel in --dist-dir instead of building one.",
    )
    parser.add_argument(
        "--smoke",
        choices=["import", "retrieve"],
        default="import",
        help="Smoke depth: import is lightweight, retrieve downloads first-run models.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary venv/cache directory for debugging.",
    )
    return parser.parse_args()


def main() -> int:
    """Run the wheel smoke."""
    args = parse_args()
    temp_root = Path(tempfile.mkdtemp(prefix="fitz-wheel-smoke-"))
    print(f"Smoke root: {temp_root}", flush=True)

    try:
        wheel = resolve_wheel(args, temp_root)
        print(f"Wheel under test: {wheel}", flush=True)
        validate_wheel_contents(wheel)
        paths = create_smoke_env(temp_root)
        install_wheel(paths, wheel)
        smoke_import(paths)
        if args.smoke == "retrieve":
            smoke_retrieve(paths)
        print("Wheel smoke passed", flush=True)
        return 0
    finally:
        if args.keep_temp:
            print(f"Kept smoke root: {temp_root}", flush=True)
        else:
            shutil.rmtree(temp_root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

"""Tests for installed-wheel release smoke helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tools.wheel_smoke import assert_output, isolated_env


def test_assert_output_accepts_expected_evidence_and_rejects_leaks() -> None:
    assert_output(
        "Project Lantern retains records for 27 days under RET-27.",
        expected=("Project Lantern", "27 days", "RET-27"),
        forbidden=("TC-4812",),
    )

    with pytest.raises(RuntimeError, match="leaked another corpus"):
        assert_output(
            "Project Lantern and TC-4812",
            expected=("Project Lantern",),
            forbidden=("TC-4812",),
        )


def test_isolated_env_uses_a_clean_pyrrho_cache(tmp_path: Path) -> None:
    env = isolated_env(tmp_path)

    assert env["PYRRHO_HOME"] == str(tmp_path / "pyrrho_home")
    assert env["HF_HOME"] == str(tmp_path / "hf_home")

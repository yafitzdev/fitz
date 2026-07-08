# tests/unit/test_benchmark_validators.py
"""Tests for retrieval benchmark validators."""

from benchmarks.fitz_bench.models import BenchmarkCase
from benchmarks.fitz_bench.validators import validate_case


def test_validate_case_matches_required_evidence() -> None:
    case = BenchmarkCase.from_dict(
        {
            "id": "case_1",
            "domain": "code",
            "query": "Where is refresh implemented?",
            "expected": {
                "mode": "sufficient",
                "required_evidence": [
                    {
                        "file": "code/auth_service.py",
                        "kind": "symbol",
                        "location_contains": "refresh_expired_session",
                        "contains": ["grace"],
                    }
                ],
            },
        }
    )
    pack = {
        "mode": "sufficient",
        "items": [
            {
                "rank": 1,
                "file_path": "benchmarks/corpora/core/code/auth_service.py",
                "source_id": "code/auth_service.py",
                "address_kind": "symbol",
                "address_location": "AuthService.refresh_expired_session",
                "excerpt": "Refresh inside the grace window.",
                "content": "return session_id.startswith('grace-')",
            }
        ],
    }

    result = validate_case(case, pack)

    assert result.passed is True
    assert result.metrics.hit_at_1 is True
    assert result.metrics.required_recall == 1.0


def test_validate_case_reports_mode_and_forbidden_failures() -> None:
    case = BenchmarkCase.from_dict(
        {
            "id": "case_2",
            "domain": "unstructured",
            "query": "What is missing?",
            "expected": {
                "mode": "insufficient",
                "forbidden_evidence": [
                    {
                        "file": "refund_policy.md",
                        "contains": ["30 days"],
                    }
                ],
            },
        }
    )
    pack = {
        "mode": "sufficient",
        "items": [
            {
                "rank": 2,
                "file_path": "unstructured/refund_policy.md",
                "source_id": "refund_policy.md",
                "address_kind": "section",
                "address_location": "Refund Policy",
                "excerpt": "Refunds within 30 days.",
                "content": "Refunds within 30 days.",
            }
        ],
    }

    result = validate_case(case, pack)

    assert result.passed is False
    assert result.metrics.mode_match is False
    assert result.metrics.forbidden_count == 1
    assert len(result.failures) == 2


def test_validate_case_accepts_v2_mode_names() -> None:
    case = BenchmarkCase.from_dict(
        {
            "id": "case_3",
            "domain": "unstructured",
            "query": "What is supported?",
            "expected": {"mode": "sufficient"},
        }
    )
    pack = {"mode": "sufficient", "items": []}

    result = validate_case(case, pack)

    assert case.expected_mode == "sufficient"
    assert result.passed is True
    assert result.metrics.mode_match is True


def test_validate_case_rejects_old_runtime_mode_names() -> None:
    case = BenchmarkCase.from_dict(
        {
            "id": "case_4",
            "domain": "unstructured",
            "query": "What is absent?",
            "expected": {"mode": "insufficient"},
        }
    )
    pack = {"mode": "abstain", "items": []}

    result = validate_case(case, pack)

    assert case.expected_mode == "insufficient"
    assert result.passed is False
    assert result.metrics.mode_match is False


def test_validate_case_matches_wrapped_markdown_phrase() -> None:
    """YAML phrases should match normal Markdown hard wraps."""
    case = BenchmarkCase.from_dict(
        {
            "id": "case_wrapped",
            "domain": "mixed",
            "query": "What does the release brief say?",
            "expected": {
                "required_evidence": [
                    {
                        "file": "mixed/release_brief.md",
                        "contains": ["roadmap status document"],
                    }
                ],
            },
        }
    )
    pack = {
        "items": [
            {
                "rank": 1,
                "file_path": "mixed/release_brief.md",
                "address_kind": "section",
                "address_location": "Release Brief",
                "excerpt": "The roadmap\nstatus document is authoritative.",
                "content": "",
            }
        ]
    }

    result = validate_case(case, pack)

    assert result.passed is True


def test_validate_case_allows_forbidden_text_inside_required_temporal_section() -> None:
    """A coarse section can contain old and current paragraphs without false failure."""
    case = BenchmarkCase.from_dict(
        {
            "id": "case_temporal_section",
            "domain": "unstructured",
            "query": "What is the current status?",
            "expected": {
                "mode": "sufficient",
                "required_evidence": [
                    {
                        "file": "status.md",
                        "contains": ["2026-08-02", "no active MFA exceptions"],
                    }
                ],
                "forbidden_evidence": [
                    {
                        "file": "status.md",
                        "contains": ["2026-03-10", "may skip hardware keys"],
                    }
                ],
            },
        }
    )
    pack = {
        "mode": "sufficient",
        "items": [
            {
                "rank": 1,
                "file_path": "status.md",
                "address_kind": "section",
                "address_location": "MFA Exceptions",
                "excerpt": "",
                "content": (
                    "2026-03-10: Pilot users may skip hardware keys.\n\n"
                    "2026-08-02: Current rule: no active MFA exceptions remain."
                ),
            }
        ],
    }

    result = validate_case(case, pack)

    assert result.passed is True
    assert result.metrics.forbidden_count == 0

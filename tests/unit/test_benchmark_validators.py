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
                "mode": "trustworthy",
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
        "mode": "trustworthy",
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
                "mode": "abstain",
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
        "mode": "trustworthy",
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

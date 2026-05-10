# tests/unit/conftest.py
"""
Test fixtures for unit tests.

Provides mock embedders for testing semantic matching without
requiring actual embedding API calls.
"""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from fitz_sage.governance import SemanticMatcher

from .mock_embedder import create_deterministic_embedder


def pytest_collection_modifyitems(config, items):
    """Add tier and postgres markers to unit tests based on type.

    Tier 1 (every commit): Pure logic tests with no I/O or mocks
    Tier 2 (PR merge): Tests with mocks but no real services
    Tier 3+: Already marked in specific files (integration, e2e)

    Postgres marker: Tests that use PostgreSQL (can't run in parallel due to pgserver)
    """
    # Files that should be tier1 (pure logic, fast, no external deps)
    # NOTE: Do NOT include postgres tests here - they can't run in parallel
    TIER1_PATTERNS = [
        "test_answer_mode",
        "test_chunker_id",
        "test_constraints",
        "test_causal_attribution",
        "test_model_tier_resolution",
        "test_query_router",
        "test_semantic_grouping",
        "test_semantic_math",
        "test_context_pipeline",
        "test_rgs",
        "test_writer_basic",
        # Tabular pure logic
        "tabular/test_models",
        "tabular/test_parser",
        # Structured pure logic
        "structured/test_types",
        "structured/test_formatter",
        "structured/test_router",
        "structured/test_schema",
        # Property-based tests (pure logic, deterministic)
        "property/",
    ]

    # Files that use PostgreSQL (pgserver) - must run serially
    POSTGRES_PATTERNS = [
        "test_pgvector",
        "test_postgres",
        "test_ingest_executor",
        "test_ingest_state",
        "test_ingest_timing",
        "test_vocabulary",
        "test_entity_graph",
        "test_retrieval_yaml_plugins",
        "test_direct_query",
        "test_vector_search_derived",
    ]

    for item in items:
        fspath = str(item.fspath)

        # Only process tests in unit directory
        if "/unit/" not in fspath and "\\unit\\" not in fspath:
            continue

        # Add postgres marker if file uses postgres
        is_postgres = any(pattern in fspath for pattern in POSTGRES_PATTERNS)
        if is_postgres:
            item.add_marker(pytest.mark.postgres)
            # Auto-skip postgres tests when running in parallel (pgserver can't handle it)
            # Check if xdist is active with multiple workers
            num_workers = getattr(config.option, "numprocesses", None)
            if num_workers is not None and num_workers != 0:
                item.add_marker(
                    pytest.mark.skip(
                        reason="Postgres tests skipped in parallel mode (run with: pytest -m postgres)"
                    )
                )

        # Skip tier marking if already has a tier marker
        has_tier = any(marker.name.startswith("tier") for marker in item.iter_markers())
        if has_tier:
            continue

        # Check if matches tier1 pattern
        is_tier1 = any(pattern in fspath for pattern in TIER1_PATTERNS)

        if is_tier1:
            item.add_marker(pytest.mark.tier1)
        else:
            item.add_marker(pytest.mark.tier2)


@pytest.fixture
def mock_embedder():
    """Fixture providing a deterministic mock embedder."""
    return create_deterministic_embedder()


@pytest.fixture
def semantic_matcher(mock_embedder) -> SemanticMatcher:
    """Fixture providing a SemanticMatcher with mock embedder."""
    return SemanticMatcher(
        # Thresholds tuned for mock embedder clusters
        causal_threshold=0.70,
        assertion_threshold=0.70,
        query_threshold=0.70,
        conflict_threshold=0.70,
    )


def _generate_test_certificate(days_valid: int = 365) -> tuple[bytes, bytes]:
    """Generate a self-signed test certificate and private key.

    Args:
        days_valid: Number of days the certificate should be valid.

    Returns:
        Tuple of (certificate_pem, private_key_pem)
    """
    # Generate private key
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    # Generate certificate
    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "US"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Test"),
            x509.NameAttribute(NameOID.COMMON_NAME, "test.example.com"),
        ]
    )

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(timezone.utc))
        .not_valid_after(datetime.now(timezone.utc) + timedelta(days=days_valid))
        .sign(private_key, hashes.SHA256())
    )

    cert_pem = cert.public_bytes(serialization.Encoding.PEM)
    key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return cert_pem, key_pem


@pytest.fixture
def temp_certificate() -> tuple[str, str]:
    """Fixture providing temporary certificate and key files.

    Returns:
        Tuple of (certificate_path, key_path) as strings.
    """
    cert_pem, key_pem = _generate_test_certificate()

    with tempfile.NamedTemporaryFile(suffix=".crt", delete=False) as cert_file:
        cert_file.write(cert_pem)
        cert_path = cert_file.name

    with tempfile.NamedTemporaryFile(suffix=".key", delete=False) as key_file:
        key_file.write(key_pem)
        key_path = key_file.name

    yield cert_path, key_path

    # Cleanup
    Path(cert_path).unlink(missing_ok=True)
    Path(key_path).unlink(missing_ok=True)

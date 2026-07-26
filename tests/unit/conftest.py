# tests/unit/conftest.py
"""Test fixtures for unit tests."""

from __future__ import annotations

import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def pytest_collection_modifyitems(config, items):
    """Add tier markers to unit tests based on type.

    Tier 1 (every commit): Pure logic tests with no I/O or mocks
    Tier 2 (PR merge): Tests with mocks but no real services
    Tier 3+: Already marked in specific files (integration, e2e)
    """
    TIER1_PATTERNS = [
        "test_answer_mode",
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

    for item in items:
        fspath = str(item.fspath)

        if "/unit/" not in fspath and "\\unit\\" not in fspath:
            continue

        has_tier = any(marker.name.startswith("tier") for marker in item.iter_markers())
        if has_tier:
            continue

        is_tier1 = any(pattern in fspath for pattern in TIER1_PATTERNS)
        if is_tier1:
            item.add_marker(pytest.mark.tier1)
        else:
            item.add_marker(pytest.mark.tier2)


@pytest.fixture
def reset_sqlite_singleton():
    """Reset the SqliteConnectionManager singleton (explicit opt-in fixture).

    Use this in tests that patch the manager class via ``@patch`` so the
    next test can rebuild a fresh real-store singleton. The fixture also
    points the SQLite storage dir at a per-test temp directory to keep
    tests fully isolated and avoid file-lock contention on Windows.
    """
    from fitz_sage.storage.sqlite import SqliteConnectionManager

    SqliteConnectionManager.reset_instance()
    yield
    SqliteConnectionManager.reset_instance()


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

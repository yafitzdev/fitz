# tests/unit/test_evidence_closure.py
"""Tests for contract-driven evidence closure planning."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.evidence_closure import (
    EvidenceClosureRequest,
    annotate_closure_result,
    plan_evidence_closure,
)
from fitz_sage.engines.fitz_krag.evidence_compiler import compile_evidence
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult
from fitz_sage.governance.evidence_contract import build_query_contract


def test_closure_plans_table_followup_from_bridge_identifier() -> None:
    """Bridge IDs discovered in selected evidence should drive table closure."""
    query = (
        "Using the operations brief, what alert maps to PAY-209 and what recovery duration "
        "did the final postmortem confirm?"
    )
    brief = _result(
        AddressKind.SECTION,
        "mixed/operations_brief.md",
        "Operations Brief",
        "Payments incident PAY-209 maps to alert ALT-501. Use the alerts table for MTTR.",
    )
    postmortem = _result(
        AddressKind.SECTION,
        "unstructured/payments_postmortem.md",
        "Final PAY-209 Postmortem",
        "The final postmortem confirmed that incident PAY-209 recovered after 37 minutes.",
    )

    profile = _profile(modality="structured_table")
    compilation = compile_evidence(query, [brief, postmortem], profile=profile)
    plan = plan_evidence_closure(query, [brief, postmortem], compilation, profile=profile)

    table_requests = [request for request in plan.requests if request.modality == "table"]
    assert table_requests
    assert table_requests[0].role == "required_table"
    assert "ALT-501" in table_requests[0].query
    assert "alerts" in table_requests[0].query


def test_closure_does_not_infer_code_documentation_companion() -> None:
    """Documentation companion searches require a Pyrrho obligation, not query regex."""
    query = "Are archived users eligible for beta feature flags?"
    symbol = _result(
        AddressKind.SYMBOL,
        "code/feature_flag_service.py",
        "FlagEvaluator.is_eligible",
        "if user.get('archived') == 'true':\n    return False",
    )

    profile = _profile(modality="code")
    compilation = compile_evidence(query, [symbol], profile=profile)
    plan = plan_evidence_closure(query, [symbol], compilation, profile=profile)

    requests = [
        request for request in plan.requests if request.reason == "code_documentation_companion"
    ]
    assert requests == []


def test_compiler_keeps_bridge_proven_table_without_literal_original_identifier() -> None:
    """Closure provenance lets companion evidence satisfy hard-anchor filtering."""
    query = (
        "Using the operations brief, what alert maps to PAY-209 and what recovery duration "
        "did the final postmortem confirm?"
    )
    brief = _result(
        AddressKind.SECTION,
        "mixed/operations_brief.md",
        "Operations Brief",
        "Payments incident PAY-209 maps to alert ALT-501.",
    )
    table = _result(
        AddressKind.TABLE,
        "structured/alerts.csv",
        "alerts",
        "Table: alerts\nALT-501,payments,P1,Imani,4,37,pager",
    )
    request = EvidenceClosureRequest(
        query="table PAY-209 ALT-501 alerts",
        modality="table",
        role="required_table",
        reason="missing_required_modality",
        bridges=("PAY-209", "ALT-501", "alerts"),
    )
    table = annotate_closure_result(
        table,
        request,
        contract=build_query_contract(query),
        run_index=1,
    )

    compilation = compile_evidence(query, [brief, table])

    selected = {
        result.file_path: result.metadata["evidence_compiler"]["roles"]
        for result in compilation.results
    }
    assert "structured/alerts.csv" in selected
    assert "required_table" in selected["structured/alerts.csv"]


def test_symbol_closure_uses_query_intent_without_table_column_pollution() -> None:
    """Symbol follow-ups should not be dominated by table schema column names."""
    query = "Which open S0 incident is in the table and which code function maps S0 to incident command?"
    table = _result(
        AddressKind.TABLE,
        "structured/incidents.csv",
        "Incidents",
        "incident_id | service_id | severity | customer_visible\nINC-730 | SVC-204 | S0 | yes",
    )

    profile = _profile(modality="code")
    compilation = compile_evidence(query, [table], profile=profile)
    plan = plan_evidence_closure(query, [table], compilation, profile=profile)
    symbol_request = next(request for request in plan.requests if request.modality == "symbol")

    assert "s0" in symbol_request.query
    assert "customer_visible" not in symbol_request.query


def test_symbol_closure_uses_observed_code_bridge_terms() -> None:
    """Code follow-ups use observed bridge terms, not generated symbol-name guesses."""
    query = "Using the export brief, should EXP-502 be skipped by the export scheduler?"
    brief = _result(
        AddressKind.SECTION,
        "mixed/export_brief.md",
        "Export Brief",
        "Export EXP-502 uses export_scheduler.py for skip logic.",
    )
    table = _result(
        AddressKind.TABLE,
        "structured/exports.csv",
        "Exports",
        "EXP-502 | analytics_cache | 840000 | no",
    )

    profile = _profile(modality="code")
    compilation = compile_evidence(query, [brief, table], profile=profile)
    plan = plan_evidence_closure(query, [brief, table], compilation, profile=profile)
    symbol_request = next(request for request in plan.requests if request.modality == "symbol")

    assert "should_skip_export" not in symbol_request.query
    assert "export_scheduler" in symbol_request.query


def test_closure_promotes_bridge_identifier_and_document_references() -> None:
    """Identifiers and document names found in an authoritative source are obligations."""
    query = (
        "Using the pricing brief, what notice applies to MeridianAI and which code path "
        "mirrors model_eval renewal notice?"
    )
    brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "MeridianAI maps to vendor VEN-301 and category model_eval. Use Procurement Policy.",
    )
    symbol = _result(
        AddressKind.SYMBOL,
        "code/pricing_engine.py",
        "pricing_engine.renewal_notice_days",
        "def renewal_notice_days(vendor_category: str) -> int: return 75",
    )

    profile = _profile(modality="structured_table")
    compilation = compile_evidence(query, [brief, symbol], profile=profile)
    plan = plan_evidence_closure(query, [brief, symbol], compilation, profile=profile)

    request_roles = {request.role: request for request in plan.requests}
    assert "bridge:VEN-301" in request_roles
    assert request_roles["bridge:VEN-301"].modality == "table"
    assert "bridge_document:procurement policy" not in request_roles


def test_compiler_carries_closure_bridge_roles() -> None:
    """Closure-proven bridge evidence should become compiler-required metadata."""
    query = "Using the release brief, which rollout and service owner apply to Project Vega EMEA?"
    service = _result(
        AddressKind.TABLE,
        "structured/services.csv",
        "Services",
        "SVC-202 | search-index | Search",
    )
    request = EvidenceClosureRequest(
        query="table SVC-202",
        modality="table",
        role="bridge:SVC-202",
        reason="bridge_identifier",
        bridges=("SVC-202",),
    )
    service = annotate_closure_result(
        service,
        request,
        contract=build_query_contract(query),
        run_index=1,
    )

    compilation = compile_evidence(query, [service])

    roles = compilation.results[0].metadata["evidence_compiler"]["roles"]
    assert "bridge:SVC-202" in roles


def test_closure_repairs_table_only_evidence_for_prose_query() -> None:
    """A non-table query should not certify table evidence without prose repair."""
    query = "What was the final outage duration confirmed for INC-611?"
    table = _result(
        AddressKind.TABLE,
        "structured/incidents.csv",
        "Incidents",
        "INC-611 | resolved | 42",
    )

    profile = _profile(modality="unstructured_text")
    compilation = compile_evidence(query, [table], profile=profile)
    plan = plan_evidence_closure(query, [table], compilation, profile=profile)

    requests = [
        request for request in plan.requests if request.reason == "missing_required_modality"
    ]
    assert requests
    assert requests[0].modality == "section"
    assert "INC-611" in requests[0].query


def _result(
    kind: AddressKind,
    file_path: str,
    location: str,
    content: str,
) -> ReadResult:
    return ReadResult(
        address=Address(
            kind=kind,
            source_id=file_path,
            location=location,
            summary=location,
            score=0.9,
            metadata={},
        ),
        content=content,
        file_path=file_path,
    )


def _profile(*, modality: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(retrieval_modality=modality)

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
from fitz_sage.engines.fitz_krag.evidence_contract import build_query_contract
from fitz_sage.engines.fitz_krag.query_pipeline import (
    _filter_companion_source_repeats,
    _merge_closure_results,
    _select_closure_results,
)
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


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


def test_bridge_identifier_followups_do_not_include_sibling_identifiers() -> None:
    """Each table bridge should retain query intent without competing bridge IDs."""
    query = "Which rollout and service owner apply to Project Vega EMEA?"
    brief = _result(
        AddressKind.SECTION,
        "mixed/release_brief.md",
        "Release Brief",
        "Project Vega EMEA is tied to rollout ROL-401 and service SVC-202.",
    )
    profile = _profile(modality="mixed")
    compilation = compile_evidence(query, [brief], profile=profile)

    plan = plan_evidence_closure(query, [brief], compilation, profile=profile)

    bridge_requests = {
        request.role: request
        for request in plan.requests
        if request.reason == "bridge_identifier"
    }
    rollout_request = bridge_requests["bridge:ROL-401"]
    service_request = bridge_requests["bridge:SVC-202"]
    assert "ROL-401" in rollout_request.query
    assert "SVC-202" not in rollout_request.query
    assert "SVC-202" in service_request.query
    assert "ROL-401" not in service_request.query
    assert "service" in service_request.query
    assert "owner" in service_request.query


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


def test_closure_does_not_expand_incidental_identifiers_from_table_rows() -> None:
    """A precise table result is endpoint evidence, not another table bridge source."""
    query = "Who owns ASSET-940?"
    table = _result(
        AddressKind.TABLE,
        "structured/assets.csv",
        "Assets",
        (
            "asset_id | owner\n"
            "ASSET-001 | Example One\n"
            "ASSET-002 | Example Two\n"
            "ASSET-940 | Target Team"
        ),
    )

    profile = _profile(modality="structured_table")
    compilation = compile_evidence(query, [table], profile=profile)
    plan = plan_evidence_closure(query, [table], compilation, profile=profile)

    assert plan.requests == []


def test_closure_does_not_follow_an_unrelated_sibling_identifier() -> None:
    """An identifier in another sentence is not a proven multi-hop bridge."""
    query = "What biometric deletion rule applies to AST-58?"
    brief = _result(
        AddressKind.SECTION,
        "mixed/compliance_brief.md",
        "Compliance Brief",
        (
            "Asset AST-22 is the analytics cache exception. "
            "Asset AST-58 handles biometric claims."
        ),
    )
    table = _result(
        AddressKind.TABLE,
        "structured/cloud_assets.csv",
        "Cloud Assets",
        "AST-58 | claims-vault | Compliance | 24",
    )
    profile = _profile(modality="mixed")
    compilation = compile_evidence(query, [brief, table], profile=profile)

    plan = plan_evidence_closure(query, [brief, table], compilation, profile=profile)

    assert not any(request.role == "bridge:AST-22" for request in plan.requests)


def test_closure_ignores_bridge_terms_from_unselected_candidates() -> None:
    """Only compiler-selected evidence may create follow-up bridge obligations."""
    query = "Who owns ASSET-940?"
    selected_table = _result(
        AddressKind.TABLE,
        "structured/assets.csv",
        "Assets",
        "ASSET-940 | Target Team",
    )
    unrelated_candidate = _result(
        AddressKind.SECTION,
        "unstructured/unrelated.md",
        "Unrelated",
        "The unrelated migration note references OTHER-777 and a records table.",
    )

    profile = _profile(modality="structured_table")
    compilation = compile_evidence(query, [selected_table], profile=profile)
    plan = plan_evidence_closure(
        query,
        [selected_table, unrelated_candidate],
        compilation,
        profile=profile,
    )

    assert plan.requests == []
    assert "OTHER-777" not in plan.metadata["bridge_terms"]


def test_required_modality_rescue_does_not_seed_an_unrelated_identifier() -> None:
    """A result kept only for modality coverage must not start a retrieval chain."""
    query = "Which legal vendor lacks SOC 2?"
    unrelated_brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "MeridianAI maps to vendor VEN-301 and category model_eval.",
    )
    correct_table = _result(
        AddressKind.TABLE,
        "structured/vendors.csv",
        "Vendors",
        "VEN-302 | Quartz Legal | legal | no",
    )
    profile = SimpleNamespace(
        retrieval_modality="mixed",
        retrieval_obligation="prose_plus_table",
        required_modalities=("section", "table"),
    )
    compilation = compile_evidence(
        query,
        [unrelated_brief, correct_table],
        profile=profile,
    )

    assert compilation.results[0].metadata["evidence_compiler"]["roles"] == [
        "required_section"
    ]

    plan = plan_evidence_closure(
        query,
        [unrelated_brief, correct_table],
        compilation,
        profile=profile,
    )

    assert "VEN-301" not in plan.metadata["bridge_terms"]
    assert all(request.role != "bridge:VEN-301" for request in plan.requests)


def test_inferred_bridge_does_not_seed_a_second_identifier_hop() -> None:
    """Compiler-inferred bridges are endpoint evidence, not new retrieval roots."""
    query = "Which alert has the shortest MTTR?"
    brief = _result(
        AddressKind.SECTION,
        "mixed/operations_brief.md",
        "Operations Brief",
        "Payments incident PAY-209 maps to alert ALT-501.",
    )
    postmortem = _result(
        AddressKind.SECTION,
        "unstructured/payments_postmortem.md",
        "Final PAY-209 Postmortem",
        "PAY-209 recovered after 37 minutes and used alert ALT-501.",
    )
    postmortem.metadata["evidence_compiler"] = {
        "roles": ["bridge:PAY-209"],
    }
    compilation = SimpleNamespace(results=[postmortem])

    plan = plan_evidence_closure(
        query,
        [brief, postmortem],
        compilation,
        profile=_profile(modality="structured_table"),
    )

    assert "ALT-501" not in plan.metadata["bridge_terms"]
    assert all(request.role != "bridge:ALT-501" for request in plan.requests)


def test_closure_follows_explicit_document_reference() -> None:
    """A real cross-document instruction should still create a bounded follow-up."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "MeridianAI renewal terms must follow Procurement Policy.",
    )
    profile = _profile(modality="unstructured_text")
    compilation = compile_evidence("What renewal terms apply?", [brief], profile=profile)

    plan = plan_evidence_closure(
        "What renewal terms apply?",
        [brief],
        compilation,
        profile=profile,
    )

    assert any(request.role == "bridge_document:procurement policy" for request in plan.requests)


def test_closure_follows_explicit_definition_for_queried_label() -> None:
    """A corpus-stated label expansion may bridge to another source."""
    query = "Who owns CBT?"
    glossary = _result(
        AddressKind.SECTION,
        "unstructured/acronym_glossary.md",
        "CBT",
        "CBT means Cell Balancing Task in this corpus.",
    )
    profile = _profile(modality="structured_table")
    compilation = compile_evidence(query, [glossary], profile=profile)

    plan = plan_evidence_closure(query, [glossary], compilation, profile=profile)

    request = next(item for item in plan.requests if item.reason == "bridge_definition")
    assert request.role == "bridge_definition:cell balancing task"
    assert request.query.startswith("Cell Balancing Task")
    assert plan.metadata["definition_bridges"] == ["Cell Balancing Task"]


def test_closure_does_not_invent_or_apply_an_unqueried_definition() -> None:
    """Definitions are followed only when their literal label occurs in the query."""
    glossary = _result(
        AddressKind.SECTION,
        "unstructured/acronym_glossary.md",
        "Glossary",
        "CBT means Cell Balancing Task. NDX means Nova Diagnostics.",
    )
    profile = _profile(modality="unstructured_text")
    compilation = compile_evidence(
        "Who owns the balancing workflow?",
        [glossary],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "Who owns the balancing workflow?",
        [glossary],
        compilation,
        profile=profile,
    )

    assert not any(item.reason == "bridge_definition" for item in plan.requests)


def test_compiler_keeps_explicit_definition_bridge_evidence() -> None:
    """Definition provenance makes the expanded source a real obligation."""
    query = "Who owns CBT?"
    glossary = _result(
        AddressKind.SECTION,
        "unstructured/acronym_glossary.md",
        "CBT",
        "CBT means Cell Balancing Task.",
    )
    ownership = _result(
        AddressKind.SECTION,
        "unstructured/system_ownership.md",
        "Cell Balancing Task",
        "Cell Balancing Task is owned by Nova Diagnostics.",
    )
    ownership = annotate_closure_result(
        ownership,
        EvidenceClosureRequest(
            query="Cell Balancing Task owns CBT",
            modality="section",
            role="bridge_definition:cell balancing task",
            reason="bridge_definition",
            bridges=("CBT", "Cell Balancing Task"),
        ),
        contract=build_query_contract(query),
        run_index=1,
    )

    compilation = compile_evidence(
        query,
        [glossary, ownership],
        profile=_profile(modality="unstructured_text"),
    )

    roles_by_path = {
        result.file_path: result.metadata["evidence_compiler"]["roles"]
        for result in compilation.results
    }
    assert (
        "bridge_definition:cell balancing task" in roles_by_path["unstructured/system_ownership.md"]
    )


def test_closure_prioritizes_explicit_document_reference_over_discovered_ids() -> None:
    """A named companion document must not lose the bounded budget to incidental IDs."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        (
            "MeridianAI maps to VEN-301, RSK-81, CTL-601, EXP-502, AST-58, and "
            "ROL-401. Use Procurement Policy for the notice requirement."
        ),
    )
    profile = SimpleNamespace(required_modalities=("section", "table"))
    compilation = compile_evidence(
        "Which notice applies to MeridianAI?",
        [brief],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "Which notice applies to MeridianAI?",
        [brief],
        compilation,
        profile=profile,
    )

    roles = [request.role for request in plan.requests]
    assert "bridge_document:procurement policy" in roles
    document_index = roles.index("bridge_document:procurement policy")
    assert plan.requests[document_index].query == "procurement policy"
    identifier_indexes = [index for index, role in enumerate(roles) if role.startswith("bridge:")]
    assert not identifier_indexes or document_index < min(identifier_indexes)


def test_document_companion_query_does_not_include_original_identifier() -> None:
    """The explicitly named document scopes its follow-up without OR-query pollution."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "For model_eval renewal terms, use Procurement Policy.",
    )
    profile = SimpleNamespace(required_modalities=("section", "symbol"))
    compilation = compile_evidence(
        "Which model_eval notice applies?",
        [brief],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "Which model_eval notice applies?",
        [brief],
        compilation,
        profile=profile,
    )

    request = next(
        item for item in plan.requests if item.role == "bridge_document:procurement policy"
    )
    assert request.query == "procurement policy"


def test_document_reference_stops_at_source_type_before_rule_description() -> None:
    """A policy reference must not absorb the descriptive rule that follows its title."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/export_brief.md",
        "Export Brief",
        "EXP-505 must follow the Data Handling Policy deletion rule.",
    )
    profile = SimpleNamespace(required_modalities=("section", "table"))
    compilation = compile_evidence(
        "Which deletion rule applies to EXP-505?",
        [brief],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "Which deletion rule applies to EXP-505?",
        [brief],
        compilation,
        profile=profile,
    )

    request = next(
        item for item in plan.requests if item.role == "bridge_document:data handling policy"
    )
    assert request.query == "data handling policy"


def test_closure_does_not_treat_source_name_as_document_reference() -> None:
    """A result's own title and path must not create self-referential closure."""
    support = _result(
        AddressKind.SECTION,
        "unstructured/support_sla.md",
        "Customer Support SLA",
        "Gold support has a 15 minute first response target for Severity 1 incidents.",
    )
    profile = _profile(modality="unstructured_text")
    compilation = compile_evidence(
        "What is the Gold support response time?",
        [support],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "What is the Gold support response time?",
        [support],
        compilation,
        profile=profile,
    )

    assert plan.requests == []


def test_closure_does_not_seed_bridges_from_residual_evidence() -> None:
    """Near-neighbor evidence cannot create obligations merely by being retrieved."""
    support = _result(
        AddressKind.SECTION,
        "unstructured/support_sla.md",
        "Customer Support SLA",
        "Gold support has a 15 minute first response target for Severity 1 incidents.",
    )
    incidents = _result(
        AddressKind.TABLE,
        "structured/incidents.csv",
        "Incidents",
        "INC-101 | S1 | west | 42",
    )
    archive = _result(
        AddressKind.SECTION,
        "archive/archive_0043.md",
        "Archived Customer Support Brief ARCH-7043",
        "Archived support notes reference ARCH-7043 and an archive_records table.",
    )
    profile = SimpleNamespace(required_modalities=("section", "table"))
    compilation = compile_evidence(
        "What is the Gold support response time for Severity 1 incidents?",
        [support, incidents, archive],
        profile=profile,
    )

    plan = plan_evidence_closure(
        "What is the Gold support response time for Severity 1 incidents?",
        [support, incidents, archive],
        compilation,
        profile=profile,
    )

    assert "ARCH-7043" not in plan.metadata["bridge_terms"]
    assert not any("ARCH-7043" in request.query for request in plan.requests)


def test_closure_does_not_read_bridge_terms_from_metadata() -> None:
    """Tracing and profile metadata are not document evidence."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/operations_brief.md",
        "Operations Brief",
        "Payments recovery follows the current incident process.",
    )
    brief.metadata["debug"] = {
        "unrelated_identifier": "ARCH-7043",
        "generated_profile": "prose_plus_table",
    }
    profile = _profile(modality="unstructured_text")
    compilation = compile_evidence("What is the recovery process?", [brief], profile=profile)

    plan = plan_evidence_closure(
        "What is the recovery process?",
        [brief],
        compilation,
        profile=profile,
    )

    assert "ARCH-7043" not in plan.metadata["bridge_terms"]
    assert "prose_plus_table" not in plan.metadata["bridge_terms"]


def test_closure_selects_one_best_grounded_followup() -> None:
    """A closure request must not certify every near-neighbor retrieval hit."""
    distractor = _result(
        AddressKind.SECTION,
        "archive/support_brief.md",
        "Archived Customer Support Brief",
        "Archived support planning notes mention incident response.",
    )
    relevant = _result(
        AddressKind.SECTION,
        "unstructured/support_sla.md",
        "Customer Support SLA",
        "Gold support has a 15 minute first response target for Severity 1 incidents.",
    )

    selected = _select_closure_results(
        "Gold support response time Severity 1 incidents",
        [relevant, distractor],
        _profile(modality="unstructured_text"),
    )

    assert [result.file_path for result in selected] == ["unstructured/support_sla.md"]


def test_closure_selects_the_requested_modality() -> None:
    """A typed closure request must not be satisfied by a stronger wrong-kind hit."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/release_brief.md",
        "Release Brief",
        "Project Vega EMEA is tied to rollout ROL-401 and service SVC-202.",
    )
    rollout = _result(
        AddressKind.TABLE,
        "structured/rollouts.csv",
        "Rollouts",
        "ROL-401 | vega_private_beta | SVC-202 | emea | 35 | 2026-09-30",
    )
    request = EvidenceClosureRequest(
        query="ROL-401 SVC-202 EMEA release table",
        modality="table",
        role="bridge:ROL-401",
        reason="bridge_identifier",
    )

    selected = _select_closure_results(
        request.query,
        [brief, rollout],
        _profile(modality="structured_table"),
        request=request,
    )

    assert len(selected) == 1
    assert selected[0].address.kind == AddressKind.TABLE
    assert selected[0].file_path == "structured/rollouts.csv"


def test_table_bridge_prefers_schema_covering_requested_fields() -> None:
    """A bridge lookup should follow the requested field, not an earlier table."""
    rollout = _result(
        AddressKind.TABLE,
        "structured/rollouts.csv",
        "Rollouts",
        "ROL-401 | vega_private_beta | SVC-202 | emea | 35",
        address_metadata={
            "name": "Rollouts",
            "columns": ["rollout_id", "service_id", "region", "rollout_percent"],
        },
    )
    service = _result(
        AddressKind.TABLE,
        "structured/services.csv",
        "Services",
        "SVC-202 | Search | Search",
        address_metadata={
            "name": "Services",
            "columns": ["service_id", "service", "region", "owner"],
        },
    )
    request = EvidenceClosureRequest(
        query="SVC-202 service owner",
        modality="table",
        role="bridge:SVC-202",
        reason="bridge_identifier",
        bridges=("SVC-202",),
    )

    selected = _select_closure_results(
        request.query,
        [rollout, service],
        _profile(modality="structured_table"),
        request=request,
    )

    assert [result.file_path for result in selected] == ["structured/services.csv"]


def test_symbol_closure_prefers_identity_covering_behavior_terms() -> None:
    """A referenced module should yield its matching function, not its wrapper."""
    module = _result(
        AddressKind.SYMBOL,
        "code/export_scheduler.py",
        "code.export_scheduler",
        '"""Export scheduling helpers."""',
        address_metadata={
            "kind": "module",
            "name": "export_scheduler",
            "qualified_name": "code.export_scheduler",
        },
    )
    function = _result(
        AddressKind.SYMBOL,
        "code/export_scheduler.py",
        "code.export_scheduler.should_skip_export",
        "def should_skip_export(row_count: int, encrypted: bool) -> bool: ...",
        address_metadata={
            "kind": "function",
            "name": "should_skip_export",
            "qualified_name": "code.export_scheduler.should_skip_export",
            "signature": "def should_skip_export(row_count: int, encrypted: bool) -> bool",
        },
    )
    request = EvidenceClosureRequest(
        query="symbol export_scheduler.py should EXP-502 be skipped",
        modality="symbol",
        role="required_symbol",
        reason="missing_required_modality",
        bridges=("EXP-502", "export_scheduler.py"),
    )

    selected = _select_closure_results(
        request.query,
        [module, function],
        _profile(modality="code"),
        request=request,
    )

    assert [result.address.location for result in selected] == [
        "code.export_scheduler.should_skip_export"
    ]


def test_document_companion_prefers_exact_source_derived_location() -> None:
    """A document section must outrank its synthetic introduction wrapper."""
    introduction = _result(
        AddressKind.SECTION,
        "unstructured/procurement_policy.md",
        "Introduction",
        "[Document: Procurement Policy]\nSubsections:\n- Procurement Policy",
    )
    policy = _result(
        AddressKind.SECTION,
        "unstructured/procurement_policy.md",
        "Procurement Policy",
        "MeridianAI requires 75 days written notice.",
    )
    request = EvidenceClosureRequest(
        query="procurement policy",
        modality="section",
        role="bridge_document:procurement policy",
        reason="bridge_document",
    )

    selected = _select_closure_results(
        request.query,
        [introduction, policy],
        _profile(modality="unstructured_text"),
        request=request,
    )

    assert [result.address.location for result in selected] == ["Procurement Policy"]


def test_document_companion_excludes_sources_already_in_evidence() -> None:
    """A companion lookup must not return the brief that requested the companion."""
    brief = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "Use Procurement Policy for the notice requirement.",
    )
    duplicate = _result(
        AddressKind.SECTION,
        "mixed/pricing_brief.md",
        "Pricing Brief",
        "Use Procurement Policy for the notice requirement.",
    )
    policy = _result(
        AddressKind.SECTION,
        "unstructured/procurement_policy.md",
        "Procurement Policy",
        "MeridianAI requires 75 days written notice.",
    )
    request = EvidenceClosureRequest(
        query="procurement policy notice",
        modality="section",
        role="bridge_document:procurement policy",
        reason="bridge_document",
    )

    filtered = _filter_companion_source_repeats(
        request,
        [brief],
        [duplicate, policy],
    )

    assert [result.file_path for result in filtered] == ["unstructured/procurement_policy.md"]


def test_table_closure_may_refresh_an_existing_table_source() -> None:
    """Table closure must retain same-source candidates for deterministic row filters."""
    table = _result(
        AddressKind.TABLE,
        "structured/vendors.csv",
        "Vendors",
        "VEN-301 | MeridianAI | 75",
    )
    request = EvidenceClosureRequest(
        query="table VEN-301",
        modality="table",
        role="bridge:VEN-301",
        reason="bridge_identifier",
    )

    filtered = _filter_companion_source_repeats(request, [table], [table])

    assert filtered == [table]


def test_closure_merge_replaces_duplicate_when_provenance_is_new() -> None:
    """A low-ranked duplicate must retain newly proven closure provenance."""
    existing = _result(
        AddressKind.SECTION,
        "unstructured/procurement_policy.md",
        "Procurement Policy",
        "MeridianAI requires 75 days written notice.",
    )
    request = EvidenceClosureRequest(
        query="procurement policy",
        modality="section",
        role="bridge_document:procurement policy",
        reason="bridge_document",
        bridges=("model_eval",),
    )
    candidate = annotate_closure_result(
        existing,
        request,
        contract=build_query_contract("Which model_eval notice applies?"),
        run_index=1,
    )

    merged, added, replaced = _merge_closure_results(
        [existing],
        [candidate],
        allow_replace=True,
    )

    assert added == 0
    assert replaced == 1
    assert merged == [candidate]


def test_closure_merge_preserves_different_equally_precise_table_result() -> None:
    """A bridge lookup must not overwrite the original query's table winner."""
    existing = _result(
        AddressKind.TABLE,
        "structured/alerts.csv",
        "Alerts",
        "ALT-504 | atlas | 19",
    )
    existing.metadata.update(
        {
            "deterministic_table_filter": True,
            "result_count": 1,
            "table_query_plan": {
                "sort": {"column": "mttr_minutes", "direction": "min"}
            },
        }
    )
    candidate = _result(
        AddressKind.TABLE,
        "structured/alerts.csv",
        "Alerts",
        "ALT-501 | payments | 37",
    )
    candidate.metadata.update(
        {
            "deterministic_table_filter": True,
            "result_count": 1,
            "evidence_closure": {"role": "bridge:ALT-501"},
        }
    )

    merged, added, replaced = _merge_closure_results(
        [existing],
        [candidate],
        allow_replace=True,
        query_contract="comparison_coverage",
    )

    assert added == 0
    assert replaced == 0
    assert merged == [existing]


def test_closure_merge_replaces_sorted_table_for_noncomparison_contract() -> None:
    """Temporal bridge evidence may replace an incidental table sort."""
    existing = _result(
        AddressKind.TABLE,
        "structured/rollouts.csv",
        "Rollouts",
        "ROL-405 | payment_replay_guard | 2027-01-15",
    )
    existing.metadata.update(
        {
            "deterministic_table_filter": True,
            "result_count": 1,
            "table_query_plan": {
                "sort": {"column": "rollout_percent", "direction": "max"}
            },
        }
    )
    candidate = _result(
        AddressKind.TABLE,
        "structured/rollouts.csv",
        "Rollouts",
        "ROL-401 | vega_private_beta | 2026-09-30",
    )
    candidate.metadata.update(
        {
            "deterministic_table_filter": True,
            "result_count": 1,
            "evidence_closure": {"role": "bridge:ROL-401"},
        }
    )

    merged, added, replaced = _merge_closure_results(
        [existing],
        [candidate],
        allow_replace=True,
        query_contract="temporal_grounding",
    )

    assert added == 0
    assert replaced == 1
    assert merged == [candidate]


def test_compiler_rejects_unproven_closure_bridge_role() -> None:
    """Closure metadata alone cannot claim an identifier absent from evidence."""
    unrelated = _result(
        AddressKind.TABLE,
        "structured/services.csv",
        "Services",
        "SVC-999 | unrelated-service | Other",
    )
    request = EvidenceClosureRequest(
        query="table SVC-202",
        modality="table",
        role="bridge:SVC-202",
        reason="bridge_identifier",
        bridges=("SVC-202",),
    )
    unrelated = annotate_closure_result(
        unrelated,
        request,
        contract=build_query_contract("Which service applies?"),
        run_index=1,
    )

    compilation = compile_evidence("Which service applies?", [unrelated])

    roles = compilation.results[0].metadata["evidence_compiler"]["roles"]
    assert "bridge:SVC-202" not in roles


def _result(
    kind: AddressKind,
    file_path: str,
    location: str,
    content: str,
    *,
    address_metadata: dict | None = None,
) -> ReadResult:
    return ReadResult(
        address=Address(
            kind=kind,
            source_id=file_path,
            location=location,
            summary=location,
            score=0.9,
            metadata=address_metadata or {},
        ),
        content=content,
        file_path=file_path,
    )


def _profile(*, modality: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(retrieval_modality=modality)

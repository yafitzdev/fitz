# tests/unit/test_evidence_compiler.py
"""Tests for contract-aware evidence compilation."""

from __future__ import annotations

from types import SimpleNamespace

from fitz_sage.engines.fitz_krag.evidence_compiler import (
    compile_evidence,
    order_addresses_for_contract,
    query_has_table_obligation,
)
from fitz_sage.engines.fitz_krag.evidence_contract import build_query_contract
from fitz_sage.engines.fitz_krag.types import Address, AddressKind, ReadResult


def test_comparison_contract_orders_companion_evidence() -> None:
    """Comparison coverage places both relevant sources first."""
    finance = _result(
        "The finance team reported Q1 revenue of 1.2 billion dollars.",
        "unstructured/finance_q1_report.md",
    )
    audit = _result(
        "The audit note disputes Q1 revenue and says it was 1.4 billion dollars.",
        "unstructured/audit_q1_note.md",
    )

    compiled = compile_evidence(
        "What was Q1 revenue?",
        [finance, audit],
        profile=_profile(query_contract="comparison_coverage"),
    )

    assert [result.file_path for result in compiled.results[:2]] == [
        "unstructured/finance_q1_report.md",
        "unstructured/audit_q1_note.md",
    ]


def test_compiler_preserves_semantic_reranker_order_without_explicit_obligation() -> None:
    """Lexical overlap must not become a second ranker after semantic reranking."""
    semantic_match = _result(
        "Refund requests are accepted within thirty days.",
        "unstructured/refund_policy.md",
    )
    lexical_match = _result(
        "Customer reimbursements are mentioned in this archived index.",
        "unstructured/archive_index.md",
    )

    compiled = compile_evidence(
        "How should customer reimbursements be handled?",
        [semantic_match, lexical_match],
    )

    assert [result.file_path for result in compiled.results] == [
        "unstructured/refund_policy.md",
        "unstructured/archive_index.md",
    ]


def test_compiler_does_not_infer_hard_anchors_from_capitalization() -> None:
    """Ordinary title-cased names must not erase reranked evidence."""
    finance = _result(
        "The finance team reported Q1 revenue of 1.2 billion dollars.",
        "unstructured/finance_q1_report.md",
    )
    audit = _result(
        "The audit note disputes Q1 revenue and says it was 1.4 billion dollars.",
        "unstructured/audit_q1_note.md",
    )

    compiled = compile_evidence("What is the Project Nebula budget?", [audit, finance])

    assert [result.file_path for result in compiled.results] == [
        "unstructured/audit_q1_note.md",
        "unstructured/finance_q1_report.md",
    ]
    assert compiled.metadata["filtered_all"] is False


def test_compiler_requires_exact_identifier_in_evidence_not_closure_metadata() -> None:
    """Closure metadata must not make unrelated text satisfy an exact identifier."""
    closure_result = _result(
        "The finance report says Q1 revenue was 1.2 billion dollars.",
        "unstructured/finance_q1_report.md",
        metadata={
            "evidence_closure": {
                "role": "bridge_document:finance report",
                "contract_identifiers": ["AX-156"],
                "bridges": ["Q1"],
            }
        },
    )

    compiled = compile_evidence("What is the AX-156 budget?", [closure_result])

    assert compiled.results == []
    assert compiled.metadata["filtered_all"] is True


def test_compiler_rejects_wrong_modality_closure_result_without_anchor() -> None:
    """A table obligation must not admit evidence missing an exact identifier."""
    closure_result = _result(
        "Project Atlas FY27 pipeline is expected to reach 7.8 million ARR.",
        "unstructured/finance_forecast.md",
        metadata={
            "evidence_closure": {
                "role": "required_table",
                "bridges": ["Project", "Helios", "budget"],
            }
        },
    )

    compiled = compile_evidence("What is the HEL-123 budget?", [closure_result])

    assert compiled.results == []
    assert compiled.metadata["filtered_all"] is True


def test_compiler_rejects_generic_bridge_overlap_without_named_entity() -> None:
    """A required table cannot replace a missing exact identifier."""
    closure_result = _result(
        "project | pipeline\nProject Atlas | 7.8 million ARR",
        "structured/projects.csv",
        kind=AddressKind.TABLE,
        metadata={
            "evidence_closure": {
                "role": "required_table",
                "bridges": ["Project", "HEL-123", "budget"],
            }
        },
    )

    compiled = compile_evidence("What is the HEL-123 budget?", [closure_result])

    assert compiled.results == []
    assert compiled.metadata["filtered_all"] is True


def test_compiler_preserves_raw_current_and_historical_spans() -> None:
    """The compiler must not rewrite a source before Pyrrho evaluates it."""
    status = _result(
        "[Introduction]\n"
        "2026-03-10: Pilot users under exception MFA-CX-13 may skip hardware keys until "
        "renewal.\n\n"
        "2026-08-02: Current rule: no active MFA exceptions remain for customer data "
        "systems.",
        "unstructured/access_policy.md",
        location="MFA Exceptions",
    )

    compiled = compile_evidence(
        "What is the current MFA exception status for customer data systems?",
        [status],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert "Current rule" in compiled.results[0].content
    assert "may skip hardware keys" in compiled.results[0].content
    assert compiled.results[0].content == status.content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_keeps_authoritative_latest_section_when_it_contains_old_history() -> None:
    """A latest section can contain older history without being suppressed as stale."""
    status = _result(
        "On 2026-03-04, Project Atlas entered a pilot in Singapore and Seoul.\n\n"
        "On 2026-07-18, Project Atlas reached limited general availability in APAC.",
        "unstructured/product_launch_status.md",
        location="Project Atlas APAC",
    )
    finance = _result(
        "Project Atlas FY27 pipeline is expected to reach 7.8 million ARR.",
        "unstructured/finance_forecast.md",
        location="Finance Forecast",
    )

    compiled = compile_evidence(
        "What is the latest APAC status for Project Atlas?",
        [finance, status],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert compiled.results[0].address.location == "Project Atlas APAC"
    assert "limited general availability" in compiled.results[0].content
    assert "entered a pilot" in compiled.results[0].content


def test_compiler_does_not_count_parser_toc_as_evidence_body() -> None:
    """File comments and child TOCs should not outrank factual subsections."""
    index = _result(
        "<!-- benchmarks/corpora/core/unstructured/project_orion_status.md -->\n\n"
        "Subsections:\n"
        "  - January Field Note\n"
        "  - May Launch Memo",
        "unstructured/project_orion_status.md",
        location="Introduction",
    )
    may = _result(
        "On 2026-05-20, Project Orion reached general availability in the EU region.",
        "unstructured/project_orion_status.md",
        location="Introduction > May Launch Memo",
    )

    compiled = compile_evidence(
        "What is the latest EU status for Project Orion?",
        [index, may],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert compiled.results[0].address.location == "Introduction > May Launch Memo"


def test_compiler_preserves_raw_estimate_and_final_spans() -> None:
    """Temporal routing may reorder evidence but must not compact its content."""
    status = _result(
        "The first status update estimated that incident PAY-209 would recover in "
        "12 minutes.\n\n"
        "The final postmortem confirmed that incident PAY-209 recovered after "
        "37 minutes.",
        "unstructured/payments_postmortem.md",
        location="PAY-209 Postmortems",
    )

    compiled = compile_evidence(
        "What was the final recovery duration for incident PAY-209?",
        [status],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert "final postmortem" in compiled.results[0].content
    assert "estimated" in compiled.results[0].content
    assert compiled.results[0].content == status.content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_does_not_focus_final_span_without_pyrrho_temporal() -> None:
    """Final/current wording alone must not trigger sidecar temporal cleanup."""
    status = _result(
        "The first status update estimated that incident PAY-209 would recover in "
        "12 minutes.\n\n"
        "The final postmortem confirmed that incident PAY-209 recovered after "
        "37 minutes.",
        "unstructured/payments_postmortem.md",
        location="PAY-209 Postmortems",
    )

    compiled = compile_evidence(
        "What was the final recovery duration for incident PAY-209?",
        [status],
    )

    assert "estimated" in compiled.results[0].content
    assert "final postmortem" in compiled.results[0].content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_keeps_non_temporal_section_proof_paragraph() -> None:
    """Focusing should not drop a second paragraph when the query is not temporal."""
    brief = _result(
        """Project Atlas APAC uses the `atlas_search` feature flag.

The semantic ranker experiment is EXP-22.""",
        "mixed/launch_brief.md",
        location="Atlas Launch Brief",
    )

    compiled = compile_evidence(
        "Which experiment is the semantic ranker experiment?",
        [brief],
    )

    assert "semantic ranker experiment is EXP-22" in compiled.results[0].content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_keeps_table_rows_when_packaging_evidence() -> None:
    """Table evidence is already structural; paragraph focusing must not drop rows."""
    table = _result(
        """Table: Cloud Assets
Columns: asset_id, service, environment, encrypted
Total rows: 5

--- Deterministic Table Matches ---
Selection: query-grounded row filter
Results (1 rows):
| asset_id | service | environment | encrypted |
| --- | --- | --- | --- |
| AST-22 | analytics-cache | prod | no |

Note: Rows selected from a bounded scan.""",
        "structured/cloud_assets.csv",
        kind=AddressKind.TABLE,
        location="Cloud Assets",
    )

    compiled = compile_evidence(
        "Which production asset is unencrypted?",
        [table],
        profile=_profile(modality="structured_table"),
    )

    assert "AST-22" in compiled.results[0].content
    assert "analytics-cache" in compiled.results[0].content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_keeps_symbol_body_fields_when_packaging_evidence() -> None:
    """Code symbols are structural evidence; focusing must not strip class fields."""
    symbol = _result(
        '''class AlertRoute:
    """Resolved routing target for an alert."""

    target: str
    notify_owner: bool''',
        "code/alert_router.py",
        kind=AddressKind.SYMBOL,
        location="code.alert_router.AlertRoute",
    )

    compiled = compile_evidence(
        "Which dataclass represents a resolved alert route?",
        [symbol],
        profile=_profile(modality="code"),
    )

    assert "target: str" in compiled.results[0].content
    assert "notify_owner: bool" in compiled.results[0].content
    assert "evidence_span" not in compiled.results[0].metadata


def test_compiler_orders_temporal_candidates_from_pyrrho_contract() -> None:
    """Temporal policy should prefer the latest equally anchored evidence first."""
    january = _result(
        "On 2026-01-15, Project Orion was still in private beta for the EU region.",
        "unstructured/project_orion_status.md",
        location="January Field Note",
    )
    may = _result(
        "On 2026-05-20, Project Orion reached general availability in the EU region.",
        "unstructured/project_orion_status.md",
        location="May Launch Memo",
    )

    compiled = compile_evidence(
        "What is the latest EU status for Project Orion?",
        [january, may],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert compiled.results[0].content.startswith("On 2026-05-20")
    assert compiled.metadata["contract"]["temporal_policy"] == "temporal"


def test_compiler_keeps_same_source_initial_estimate_and_final_evidence() -> None:
    """Pyrrho, rather than the compiler, judges which temporal fact governs."""
    final = _result(
        "The final postmortem confirmed that Search outage INC-101 recovered after 42 minutes.",
        "unstructured/outage_postmortem.md",
        location="Final Postmortem",
    )
    initial = _result(
        "The initial status update estimated that Search outage INC-101 would recover in 20 minutes.",
        "unstructured/outage_postmortem.md",
        location="Initial Status Update",
    )

    compiled = compile_evidence(
        "What was the final recovery duration for Search outage INC-101?",
        [initial, final],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert [result.content for result in compiled.results] == [final.content, initial.content]


def test_compiler_keeps_cross_source_stale_and_final_evidence() -> None:
    """The evidence pack must preserve disagreement for Pyrrho to evaluate."""
    final = _result(
        "Risk RSK-81 final residual score is 18 after reclassification.",
        "unstructured/risk_audit_note.md",
        location="Audit Risk Note",
    )
    stale = _result(
        "Risk RSK-81 residual score was listed as 12 before remediation.",
        "unstructured/risk_finance_memo.md",
        location="Finance Risk Memo",
    )

    compiled = compile_evidence(
        "What is the final residual score for risk RSK-81?",
        [final, stale],
        profile=_profile(query_contract="temporal_grounding"),
    )

    assert [result.file_path for result in compiled.results] == [
        "unstructured/risk_audit_note.md",
        "unstructured/risk_finance_memo.md",
    ]


def test_compiler_promotes_required_code_symbol() -> None:
    """An explicit Python symbol query should put symbol evidence before stale prose."""
    stale = _result(
        "The authentication documentation says expired sessions are never refreshed.",
        "code/README.md",
        kind=AddressKind.SECTION,
        location="Authentication Notes",
    )
    symbol = _result(
        "def refresh_expired_session(self, session_id: str) -> bool:\n"
        "    return session_id.startswith('grace-')",
        "code/auth_service.py",
        kind=AddressKind.SYMBOL,
        location="AuthService.refresh_expired_session",
    )

    compiled = compile_evidence(
        "Which Python symbol implements expired session refresh inside the grace window?",
        [stale, symbol],
        profile=_profile(modality="code"),
    )

    assert compiled.results[0].file_path == "code/auth_service.py"
    assert compiled.metadata["contract"]["required_modalities"] == ["symbol"]


def test_address_rescue_preserves_required_table_candidate() -> None:
    """Required table candidates dropped by reranking should be appended before read."""
    prose = _address(
        AddressKind.SECTION,
        "mixed/release_notes.md",
        "Release Notes",
        "Release notes mention 17 flux capacitor units.",
    )
    table = _address(
        AddressKind.TABLE,
        "structured/warehouses.csv",
        "Warehouses",
        "Table Warehouses columns: warehouse_id, region, item, stock, unit.",
    )

    ordered = order_addresses_for_contract(
        "How many flux capacitor units are in the west region?",
        [prose, table],
        [prose],
        profile=_profile(modality="structured_table"),
    )

    assert ordered == [prose, table]
    assert query_has_table_obligation(
        "How many flux capacitor units are in the west region?",
        profile=_profile(modality="structured_table"),
    )
    assert not query_has_table_obligation("How many flux capacitor units are in the west region?")


def test_address_rescue_uses_profile_obligation_for_companion_table() -> None:
    """A prose-plus-table Pyrrho obligation should preserve both evidence kinds."""
    prose = _address(
        AddressKind.SECTION,
        "mixed/security_rollout.md",
        "Security Rollout Brief",
        "The EU token rotation rollout follows the Security Policy.",
    )
    table = _address(
        AddressKind.TABLE,
        "structured/rollout_matrix.csv",
        "Rollout Matrix",
        "Table Rollout Matrix columns: feature, region, status, release.",
    )

    ordered = order_addresses_for_contract(
        "Which EU token rotation release and policy interval apply?",
        [prose, table],
        [prose],
        profile=_profile(modality="unstructured_text", obligation="prose_plus_table"),
    )

    assert ordered == [prose, table]


def test_address_rescue_replaces_duplicate_inside_full_read_window() -> None:
    """Required evidence must not be appended beyond the reranker's read limit."""
    distractors = [
        _address(
            AddressKind.SECTION,
            f"archive_{index}.md",
            f"Archive {index}",
            "Archived warehouse operations brief.",
        )
        for index in range(10)
    ]
    table = _address(
        AddressKind.TABLE,
        "structured/warehouses.csv",
        "Warehouses",
        "Table Warehouses columns: warehouse_id, region, item, stock, unit.",
    )

    ordered = order_addresses_for_contract(
        "How many flux capacitor units are in the west region?",
        [*distractors, table],
        distractors,
        profile=_profile(modality="structured_table"),
        limit=10,
    )

    assert len(ordered) == 10
    assert ordered[-1] == table


def test_address_rescue_does_not_add_second_best_required_candidate() -> None:
    """An already-selected best candidate satisfies its required modality."""
    weak_table = _address(
        AddressKind.TABLE,
        "structured/incidents.csv",
        "Incidents",
        "Table Incidents columns: incident_id, service.",
    )
    best_table = _address(
        AddressKind.TABLE,
        "structured/warehouses.csv",
        "Warehouses",
        "Table Warehouses columns: warehouse_id, region, item, stock, unit.",
    )

    ordered = order_addresses_for_contract(
        "How many flux capacitor units are in the west region?",
        [weak_table, best_table],
        [best_table],
        profile=_profile(modality="structured_table"),
    )

    assert ordered == [best_table]


def test_address_rescue_preserves_one_rare_literal_bm25_hit() -> None:
    """A direct rare query-token hit must remain readable after reranking."""
    glossary = _address(
        AddressKind.SECTION,
        "unstructured/glossary.md",
        "CBT",
        "CBT means Cell Balancing Task.",
    )
    table = _address(
        AddressKind.TABLE,
        "structured/records.csv",
        "Records",
        "Table Records columns: record_id, owner.",
    )
    distractors = [
        _address(
            AddressKind.SECTION,
            f"unstructured/owner_{index}.md",
            f"Owner {index}",
            "Ownership details for another system.",
        )
        for index in range(10)
    ]

    ordered = order_addresses_for_contract(
        "Who owns CBT?",
        [glossary, table, *distractors],
        [table, *distractors[:9]],
        profile=_profile(modality="structured_table"),
        limit=10,
    )

    assert len(ordered) == 10
    assert glossary in ordered
    assert table in ordered


def test_query_contract_does_not_make_question_prefix_an_entity() -> None:
    """Question wording should not become a hard phrase anchor."""
    contract = build_query_contract(
        "Which Python symbol implements expired session refresh inside the grace window?"
    )

    assert contract.identifiers == ()


def test_compiler_does_not_treat_question_auxiliary_as_entity_anchor() -> None:
    """Question auxiliaries such as 'Do Acme' must not filter aligned evidence."""
    current = _result(
        "Current Acme Support refund policy: customers can request refunds within 30 days.",
        "unstructured/refund_policy.md",
    )
    legacy = _result(
        "Archived Acme refund note from 2021: customers could request refunds within 14 days.",
        "unstructured/legacy_refund_note.md",
    )

    compiled = compile_evidence(
        "Do Acme refund notes agree on the refund window?",
        [legacy, current],
    )

    assert compiled.metadata.get("filtered_all") is not True
    assert [result.file_path for result in compiled.results] == [
        "unstructured/legacy_refund_note.md",
        "unstructured/refund_policy.md",
    ]


def test_compiler_does_not_rerank_code_symbols_by_keyword_overlap() -> None:
    """Symbol-name overlap must not replace the cross-encoder's ordering."""
    constant = _result(
        'REQUIRED_ENV_VARS = ("FITZ_API_TOKEN", "FITZ_WORKSPACE")',
        "code/config_loader.py",
        kind=AddressKind.SYMBOL,
        location="code.config_loader.REQUIRED_ENV_VARS",
    )
    function = _result(
        "def load_required_env() -> dict[str, str]:\n"
        "    return {name: os.environ[name] for name in REQUIRED_ENV_VARS}",
        "code/config_loader.py",
        kind=AddressKind.SYMBOL,
        location="code.config_loader.load_required_env",
    )

    compiled = compile_evidence(
        "Which required environment variable stores the API token?",
        [function, constant],
    )

    assert [result.address.location for result in compiled.results] == [
        "code.config_loader.load_required_env",
        "code.config_loader.REQUIRED_ENV_VARS",
    ]


def test_compiler_treats_short_letter_digit_codes_as_identifiers() -> None:
    """Codes such as S1 are exact anchors, not disposable short words."""
    table = _result(
        "incident_id | severity | owner | resolved_minutes\nINC-103 | S1 | Mina | 25",
        "structured/incidents.csv",
        kind=AddressKind.TABLE,
        location="Incidents",
    )
    prose = _result(
        "The final postmortem confirmed Search outage INC-101 recovered after 42 minutes.",
        "unstructured/outage_postmortem.md",
    )

    compiled = compile_evidence(
        "Which S1 incident had the shortest resolution time?",
        [prose, table],
    )

    assert compiled.results[0].file_path == "structured/incidents.csv"
    assert compiled.metadata["contract"]["identifiers"] == ["S1"]


def test_compiler_does_not_invent_source_authority_roles() -> None:
    """Source-authority roles are not inferred from 'using/from' query wording."""
    playbook = _result(
        "The west-region cache replay fix came from Search outage INC-101.",
        "mixed/incident_playbook.md",
        location="Incident Playbook",
    )
    table = _result(
        "incident_id | owner | resolved_minutes\nINC-101 | Nora | 42",
        "structured/incidents.csv",
        kind=AddressKind.TABLE,
        location="Incidents",
    )
    postmortem = _result(
        "The final postmortem confirmed that Search outage INC-101 recovered after 42 minutes.",
        "unstructured/outage_postmortem.md",
    )

    compiled = compile_evidence(
        "Using the incident playbook, who owned the west-region cache replay fix "
        "and how long did INC-101 take to recover?",
        [postmortem, table, playbook],
    )

    assert compiled.results[0].file_path == "unstructured/outage_postmortem.md"
    assert (
        "source_anchor:incident playbook"
        not in compiled.results[0].metadata["evidence_compiler"]["roles"]
    )


def test_compiler_preserves_top_ranked_table_when_modality_is_already_satisfied() -> None:
    """A satisfied modality must not trigger lexical reranking among tables."""
    rollout = _result(
        "feature | region | status | release | owner\n"
        "warehouse_audit | west | enabled | 2026.05 | Inventory",
        "structured/rollout_matrix.csv",
        kind=AddressKind.TABLE,
        location="Rollout Matrix",
    )
    warehouses = _result(
        "warehouse_id | region | item | stock | unit\nWH-1 | west | flux capacitor | 17 | count",
        "structured/warehouses.csv",
        kind=AddressKind.TABLE,
        location="Warehouses",
    )
    release = _result(
        "Release 2026.05 added the west-region warehouse audit and confirmed "
        "17 flux capacitor units in warehouse WH-1.",
        "mixed/release_notes.md",
        location="Release Notes",
    )

    compiled = compile_evidence(
        "Which release mentioned the west-region warehouse audit and how many "
        "flux capacitor units did it confirm?",
        [release, rollout, warehouses],
        profile=_profile(modality="structured_table"),
    )

    assert compiled.results[0].file_path == "structured/rollout_matrix.csv"
    assert compiled.results[0].metadata["evidence_compiler"]["roles"] == ["required_table"]


def test_compiler_preserves_reranker_order_among_exact_identifier_matches() -> None:
    """Exact identifiers filter candidates but do not create a second ranking signal."""
    note = _result(
        "Incident INC-103 appears in the operations status note.",
        "unstructured/status.md",
        kind=AddressKind.SECTION,
        location="Status Note",
    )
    table = _result(
        "incident_id | service | owner\nINC-103 | auth | Mina",
        "structured/incidents.csv",
        kind=AddressKind.TABLE,
        location="Incidents",
    )

    compiled = compile_evidence(
        "Who is the owner for incident INC-103?",
        [note, table],
        profile=_profile(obligation="error_signature"),
    )

    assert compiled.results[0].file_path == "unstructured/status.md"
    assert compiled.results[0].metadata["evidence_compiler"]["roles"] == [
        "anchor_identifier:INC-103",
        "required_section",
    ]


def test_compiler_promotes_bridge_companion_from_pyrrho_multi_modality() -> None:
    """Bridge IDs in selected prose should pull tables only under Pyrrho companion need."""
    postmortem = _result(
        "Final PAY-209 postmortem confirms alert ALT-501. Use the alerts table for duration.",
        "unstructured/payments_postmortem.md",
        location="Final PAY-209 Postmortem",
    )
    alerts = _result(
        """Table: Alerts
Columns: alert_id, duration_minutes

| alert_id | duration_minutes |
| --- | --- |
| ALT-501 | 37 |""",
        "structured/alerts.csv",
        kind=AddressKind.TABLE,
        location="Alerts",
    )

    compiled = compile_evidence(
        "Which alert maps to PAY-209 and what was the final duration?",
        [postmortem, alerts],
        profile=_profile(obligation="prose_plus_table"),
    )

    assert [result.file_path for result in compiled.results[:2]] == [
        "unstructured/payments_postmortem.md",
        "structured/alerts.csv",
    ]
    assert "bridge:ALT-501" in compiled.results[1].metadata["evidence_compiler"]["roles"]


def test_compiler_keeps_validated_document_companion_without_original_identifier() -> None:
    """An explicit source bridge may lead to a policy that does not repeat the row ID."""
    brief = _result(
        "For model_eval renewal terms, use Procurement Policy.",
        "mixed/pricing_brief.md",
        location="Pricing Brief",
    )
    policy = _result(
        "Procurement Policy: MeridianAI requires 75 days written notice.",
        "unstructured/procurement_policy.md",
        location="Procurement Policy",
        metadata={
            "evidence_closure": {
                "role": "bridge_document:procurement policy",
                "bridges": ["model_eval", "Procurement Policy"],
                "contract_identifiers": ["model_eval"],
            }
        },
    )

    compiled = compile_evidence(
        "Which model_eval notice applies?",
        [brief, policy],
    )

    assert [result.file_path for result in compiled.results] == [
        "mixed/pricing_brief.md",
        "unstructured/procurement_policy.md",
    ]
    assert compiled.results[1].metadata["evidence_compiler"]["roles"] == [
        "bridge_document:procurement policy"
    ]


def test_compiler_does_not_promote_bridge_companion_without_pyrrho_multi_modality() -> None:
    """Bridge companion pulls are not fitz-sage-owned when Pyrrho asks for one modality."""
    postmortem = _result(
        "Final PAY-209 postmortem confirms alert ALT-501. Use the alerts table for duration.",
        "unstructured/payments_postmortem.md",
        location="Final PAY-209 Postmortem",
    )
    alerts = _result(
        "ALT-501 | 37",
        "structured/alerts.csv",
        kind=AddressKind.TABLE,
        location="Alerts",
    )

    compiled = compile_evidence(
        "Which alert maps to PAY-209 and what was the final duration?",
        [postmortem, alerts],
    )

    assert [result.file_path for result in compiled.results] == [
        "unstructured/payments_postmortem.md"
    ]


def test_compiler_does_not_make_policy_terms_source_authority() -> None:
    """Policy wording is lexical alignment unless Pyrrho supplies an authority signal."""
    table = _result(
        "feature | region | status | release\ntoken_rotation | eu | enabled | 2026.05",
        "structured/rollout_matrix.csv",
        kind=AddressKind.TABLE,
        location="Rollout Matrix",
    )
    brief = _result(
        "The token rotation rollout follows the Security Policy rotation interval. "
        "The policy interval remains 45 days.",
        "mixed/security_rollout.md",
        location="Security Rollout Brief",
    )
    policy = _result(
        "Security Policy: Service tokens must rotate every 45 days.",
        "unstructured/security_policy.md",
        location="Security Policy",
    )

    compiled = compile_evidence(
        "For the EU token rotation rollout, which release enabled it and what "
        "policy interval applies?",
        [table, brief, policy],
    )

    roles = [
        role
        for result in compiled.results
        for role in result.metadata["evidence_compiler"]["roles"]
    ]
    assert not any(role.startswith("source_anchor:") for role in roles)


def test_query_contract_does_not_force_table_for_generic_incident() -> None:
    """Incident prose should not become a table obligation without row/table wording."""
    contract = build_query_contract("Who owns the privacy incident response process?")

    assert contract.required_modalities == ()


def test_query_contract_requires_table_only_from_pyrrho_modality() -> None:
    """Schema wording does not create table obligations without Pyrrho."""
    contract = build_query_contract(
        "Which service has the highest SLO percent?",
        profile=_profile(modality="structured_table"),
    )

    assert contract.required_modalities == ("table",)
    assert (
        build_query_contract("Which service has the highest SLO percent?").required_modalities == ()
    )


def test_query_contract_requires_modalities_from_profile_obligation() -> None:
    """Pyrrho retrieval_obligation is part of the evidence contract."""
    contract = build_query_contract(
        "For the EU token rotation rollout, which release enabled it?",
        profile=_profile(modality="unstructured_text", obligation="prose_plus_table"),
    )

    assert contract.retrieval_obligation == "prose_plus_table"
    assert contract.required_modalities == ("section", "table")


def test_query_contract_mixed_modality_requires_mixed_evidence_coverage() -> None:
    """Pyrrho mixed modality should not collapse to a single evidence kind."""
    contract = build_query_contract(
        "Using the billing brief, which vendor owns INV-702 and which code function calculates it?",
        profile=_profile(modality="mixed"),
    )

    assert contract.required_modalities == ("section", "table", "symbol")


def test_query_contract_obligation_refines_coarse_mixed_modality() -> None:
    """A precise mixed obligation should not require an unrelated third kind."""
    contract = build_query_contract(
        "Using the implementation guide, which function handles authentication?",
        profile=_profile(modality="mixed", obligation="prose_plus_code"),
    )

    assert contract.required_modalities == ("section", "symbol")


def test_query_contract_keeps_code_export_window_as_symbol_only() -> None:
    """Code symbol questions should not inherit table obligations from domain nouns."""
    contract = build_query_contract(
        "Which function returns the APAC export window of 16:00?",
        profile=_profile(modality="code"),
    )

    assert contract.required_modalities == ("symbol",)


def test_query_contract_treats_vendor_ids_as_structured_identifiers() -> None:
    """Structured identifiers beyond the starter corpus should imply row evidence."""
    contract = build_query_contract(
        "What notice days are recorded for vendor VEN-301?",
        profile=_profile(modality="structured_table"),
    )

    assert contract.identifiers == ("VEN-301",)
    assert "table" in contract.required_modalities


def test_query_contract_does_not_force_table_for_incident_postmortem_fact() -> None:
    """Incident identifiers can be prose anchors unless row evidence is requested."""
    contract = build_query_contract("What final outage duration was confirmed for INC-611?")

    assert contract.identifiers == ("INC-611",)
    assert contract.required_modalities == ()


def test_compiler_needs_explicit_temporal_contract_to_reorder_candidates() -> None:
    """Temporal wording alone does not bypass the trusted query profile."""
    deprecation = _result(
        "The legacy_sync deprecation date is 2026-09-30.",
        "unstructured/project_atlas_status.md",
        location="Project Atlas Deprecations",
    )
    apac = _result(
        "On 2026-07-18, Project Atlas APAC moved to limited GA.",
        "unstructured/project_atlas_status.md",
        location="APAC Status",
    )

    compiled = compile_evidence(
        "What is the latest APAC status for Project Atlas?",
        [deprecation, apac],
    )

    assert compiled.results[0].content.startswith("The legacy_sync")
    assert any("legacy_sync" in result.content for result in compiled.results)


def test_compiler_preserves_reranker_order_across_symbol_granularities() -> None:
    """Compiler keyword heuristics must not choose between class and method results."""
    class_symbol = _result(
        "class FeatureGate:\n    def is_eligible(self, account): ...",
        "code/flags.py",
        kind=AddressKind.SYMBOL,
        location="flags.FeatureGate",
        address_metadata={
            "kind": "class",
            "name": "FeatureGate",
            "qualified_name": "flags.FeatureGate",
        },
    )
    method_symbol = _result(
        "def is_eligible(self, account):\n    return account.flag_enabled",
        "code/flags.py",
        kind=AddressKind.SYMBOL,
        location="flags.FeatureGate.is_eligible",
        address_metadata={
            "kind": "method",
            "name": "is_eligible",
            "qualified_name": "flags.FeatureGate.is_eligible",
        },
    )

    compiled = compile_evidence(
        "Which method handles feature flag eligibility?",
        [class_symbol, method_symbol],
    )

    assert compiled.results[0].address.location == "flags.FeatureGate"


def test_behavioral_code_query_preserves_code_and_documentation() -> None:
    """Fitz keeps both source types available for Pyrrho's judgment."""
    symbol = _result(
        "if user.get('archived') == 'true':\n    return False",
        "code/feature_flag_service.py",
        kind=AddressKind.SYMBOL,
        location="code.feature_flag_service.FlagEvaluator.is_eligible",
        address_metadata={
            "kind": "method",
            "name": "is_eligible",
            "qualified_name": "code.feature_flag_service.FlagEvaluator.is_eligible",
        },
    )
    notes = _result(
        "The stale feature-flag note says archived users remain eligible for beta flags.",
        "code/README.md",
        location="Feature Flags",
    )

    compiled = compile_evidence(
        "Are archived users eligible for beta feature flags?",
        [symbol, notes],
        profile=_profile(modality="code"),
    )

    roles_by_file = {
        result.file_path: result.metadata["evidence_compiler"]["roles"]
        for result in compiled.results
    }
    assert "required_symbol" in roles_by_file["code/feature_flag_service.py"]
    assert "code/README.md" in roles_by_file


def test_code_notes_export_claims_are_both_preserved() -> None:
    """Opposing code/prose claims remain raw evidence for Pyrrho."""
    symbol = _result(
        'if user.get("type") == "guest":\n    return False',
        "code/access_control.py",
        kind=AddressKind.SYMBOL,
        location="code.access_control.AccessPolicy.can_export_data",
        address_metadata={
            "kind": "method",
            "name": "can_export_data",
            "qualified_name": "code.access_control.AccessPolicy.can_export_data",
        },
    )
    notes = _result(
        "The legacy access guide says guest collaborators can export Amber project data.",
        "code/README.md",
        location="Holdout2 Code Notes",
    )

    compiled = compile_evidence(
        "Can guest collaborators export Amber project data according to the code and code notes?",
        [symbol, notes],
        profile=_profile(modality="code"),
    )

    assert {result.file_path for result in compiled.results} == {
        "code/access_control.py",
        "code/README.md",
    }


def _profile(
    *,
    query_contract: str | None = None,
    modality: str | None = None,
    obligation: str | None = None,
    shape: str | None = None,
) -> SimpleNamespace:
    """Build a Pyrrho-derived profile fixture."""
    return SimpleNamespace(
        query_contract=query_contract,
        retrieval_modality=modality,
        retrieval_obligation=obligation,
        answerability_shape=shape,
    )


def _result(
    content: str,
    file_path: str,
    *,
    kind: AddressKind = AddressKind.SECTION,
    location: str = "Section",
    address_metadata: dict | None = None,
    metadata: dict | None = None,
) -> ReadResult:
    """Build a read result fixture."""
    address = Address(
        kind=kind,
        source_id=file_path,
        location=location,
        summary=content,
        score=1.0,
        metadata={"source_path": file_path, **(address_metadata or {})},
    )
    return ReadResult(
        address=address, content=content, file_path=file_path, metadata=metadata or {}
    )


def _address(
    kind: AddressKind,
    source_id: str,
    location: str,
    summary: str,
) -> Address:
    """Build an address fixture."""
    return Address(
        kind=kind,
        source_id=source_id,
        location=location,
        summary=summary,
        score=1.0,
        metadata={"source_path": source_id},
    )

# fitz_sage/engines/fitz_krag/evidence_compiler.py
"""Evidence contract compilation for KRAG evidence packs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fitz_sage.engines.fitz_krag.evidence_contract import (
    EXACT_IDENTIFIER_PATTERN as _EXACT_IDENTIFIER_PATTERN,
)
from fitz_sage.engines.fitz_krag.evidence_contract import QueryContract as _QueryContract
from fitz_sage.engines.fitz_krag.evidence_contract import (
    build_query_contract as _build_query_contract,
)
from fitz_sage.engines.fitz_krag.evidence_contract import (
    contains_exact_identifier as _contains_exact_identifier,
)
from fitz_sage.engines.fitz_krag.evidence_contract import normalize_text as _normalize_text
from fitz_sage.engines.fitz_krag.types import Address, ReadResult

_NUMBER_PATTERN = re.compile(
    r"\b\d+(?:\.\d+)?(?:\s*(?:billion|million|percent|minutes?|days?|hours?))?\b",
    re.IGNORECASE,
)
_ISO_DATE_PATTERN = re.compile(r"\b(20\d{2})-(\d{2})-(\d{2})\b")


@dataclass(frozen=True)
class EvidenceUnit:
    """Normalized view of one read result or address candidate."""

    index: int
    kind: str
    file_path: str
    location: str
    text: str
    content_text: str
    result: ReadResult | None = None
    address: Address | None = None
    numbers: tuple[str, ...] = ()
    alignment_score: int = 0
    roles: tuple[str, ...] = ()


@dataclass(frozen=True)
class EvidenceCompilation:
    """Compiled evidence and trace metadata."""

    results: list[ReadResult]
    metadata: dict[str, Any] = field(default_factory=dict)


def compile_evidence(
    query: str,
    results: list[ReadResult],
    profile: Any = None,
) -> EvidenceCompilation:
    """Compile ranked read results into an evidence-contract order."""
    contract = _build_query_contract(query, profile)
    if not results:
        return EvidenceCompilation([], _metadata(contract, [], [], 0))

    units = [_read_result_unit(index, result, contract) for index, result in enumerate(results)]
    aligned = _aligned_units(contract, units)
    if contract.has_hard_anchors and not aligned:
        return EvidenceCompilation([], _metadata(contract, units, [], 0, filtered_all=True))

    working = aligned if aligned else units
    ordered = _compile_order(contract, working, all_units=units)
    # Preserve retrieved evidence verbatim. Query-shape logic may reorder evidence,
    # but authority resolution and conflict judgment belong to Pyrrho and the caller.
    suppressed: list[EvidenceUnit] = []
    min_sources = _minimum_sources(contract, ordered)
    compiled: list[ReadResult] = []
    for rank, unit in enumerate(ordered, start=1):
        result = unit.result
        if result is not None:
            compiled.append(_with_compiler_metadata(result, unit, rank, min_sources, contract))
    return EvidenceCompilation(
        compiled,
        _metadata(contract, units, ordered, min_sources, suppressed=suppressed),
    )


def order_addresses_for_contract(
    query: str,
    candidates: list[Address],
    selected: list[Address],
    profile: Any = None,
    *,
    limit: int | None = None,
) -> list[Address]:
    """Preserve the best Pyrrho-required candidates inside the read window."""
    contract = _build_query_contract(query, profile)
    if not candidates or not _needs_candidate_rescue(contract):
        return selected

    selected_keys = {(address.source_id, address.location) for address in selected}
    rescued: list[Address] = []
    candidate_units = [
        _address_unit(index, address, contract) for index, address in enumerate(candidates)
    ]
    for modality in contract.required_modalities:
        matches = [
            unit for unit in candidate_units if unit.kind == modality and unit.address is not None
        ]
        if not matches:
            continue
        match = _best_modality_unit(contract, matches)
        if match is None or match.address is None:
            continue
        if (match.address.source_id, match.address.location) in selected_keys:
            continue
        selected_keys.add((match.address.source_id, match.address.location))
        rescued.append(match.address)

    if not rescued:
        return selected
    return _merge_rescued_addresses(
        selected,
        rescued,
        contract.required_modalities,
        limit=limit,
    )


def _merge_rescued_addresses(
    selected: list[Address],
    rescued: list[Address],
    required_modalities: tuple[str, ...],
    *,
    limit: int | None,
) -> list[Address]:
    """Replace weak duplicate kinds before expanding a reranker's read window."""
    merged = list(selected)
    protected_kinds = set(required_modalities)

    for rescue in rescued:
        rescue_key = (rescue.source_id, rescue.location)
        if any((item.source_id, item.location) == rescue_key for item in merged):
            continue
        if limit is None or len(merged) < limit:
            merged.append(rescue)
            continue

        counts: dict[str, int] = {}
        for item in merged:
            kind = getattr(item.kind, "value", str(item.kind))
            counts[kind] = counts.get(kind, 0) + 1

        replace_at = next(
            (
                index
                for index in range(len(merged) - 1, -1, -1)
                if counts.get(
                    getattr(merged[index].kind, "value", str(merged[index].kind)),
                    0,
                )
                > (
                    1
                    if getattr(
                        merged[index].kind,
                        "value",
                        str(merged[index].kind),
                    )
                    in protected_kinds
                    else 0
                )
            ),
            None,
        )
        if replace_at is not None:
            merged[replace_at] = rescue
    return merged


def query_has_table_obligation(query: str, profile: Any = None) -> bool:
    """Return whether Pyrrho requires table evidence."""
    contract = _build_query_contract(query, profile)
    return "table" in contract.required_modalities


def query_has_code_obligation(query: str, profile: Any = None) -> bool:
    """Return whether Pyrrho requires code-symbol evidence."""
    contract = _build_query_contract(query, profile)
    return "symbol" in contract.required_modalities


def _compile_order(
    contract: _QueryContract,
    units: list[EvidenceUnit],
    *,
    all_units: list[EvidenceUnit] | None = None,
) -> list[EvidenceUnit]:
    """Build evidence order from Pyrrho obligations and literal alignment."""
    search_units = all_units or units
    selected: list[EvidenceUnit] = []
    selected_positions: dict[tuple[str, str, str], int] = {}

    def add(unit: EvidenceUnit, role: str) -> None:
        key = (unit.kind, unit.file_path, unit.location)
        if key in selected_positions:
            if role in {"aligned", "residual"}:
                return
            index = selected_positions[key]
            selected[index] = _with_role(selected[index], role)
            return
        selected_positions[key] = len(selected)
        selected.append(_with_role(unit, role))

    for identifier in contract.identifiers:
        identifier_units = [
            unit
            for unit in units
            if _contains_identifier(_unit_text(unit.content_text, unit.location), identifier)
        ]
        match = _best_unit(contract, identifier_units)
        if match is not None:
            add(match, f"anchor_identifier:{identifier}")

    for phrase in contract.phrase_anchors:
        phrase_units = [unit for unit in units if _contains_phrase(unit.text, phrase)]
        match = _best_unit(contract, phrase_units)
        if match is not None:
            add(match, f"anchor_phrase:{phrase}")

    for unit in units:
        for role in unit.roles:
            if _role_is_contract_obligation(role):
                add(unit, role)

    for modality in contract.required_modalities:
        match = _best_modality_unit(
            contract,
            [unit for unit in units if unit.kind == modality],
        )
        if match is not None:
            add(match, f"required_{modality}")

    for unit, term in _bridge_companion_units(contract, search_units, selected):
        add(unit, f"bridge:{term}")

    for unit in sorted(
        units,
        key=lambda item: _unit_order_key(contract, item),
    ):
        add(unit, "aligned" if unit.alignment_score > 0 else "residual")

    return selected


def _aligned_units(contract: _QueryContract, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Return units that satisfy literal anchors when anchors exist."""
    if contract.identifiers or contract.phrase_anchors:
        return [
            unit
            for unit in units
            if _identity_hard_anchor_score(contract, unit) > 0
            or _closure_bridge_content_score(unit) > 0
        ]
    if contract.keyword_anchors:
        return [unit for unit in units if unit.alignment_score > 0]
    return units


def _minimum_sources(contract: _QueryContract, units: list[EvidenceUnit]) -> int:
    """Return how many compiled sources Pyrrho should inspect before trusting."""
    if not units:
        return 0
    required = 1
    required = max(required, len(contract.required_modalities))
    if contract.query_contract == "comparison_coverage":
        required = max(required, 2)
    if contract.query_contract == "exhaustive_coverage":
        required = max(required, 3)
    if contract.answerability_shape == "set_answer":
        required = max(required, 3)
    if contract.retrieval_modality == "mixed":
        required = max(required, 2)
    floor_roles = {
        role
        for unit in units
        for role in unit.roles
        if _role_requires_prefix_floor(role, unit.kind)
    }
    required = max(required, len(floor_roles))
    return min(required, len(units))


def _read_result_unit(index: int, result: ReadResult, contract: _QueryContract) -> EvidenceUnit:
    """Build a normalized evidence unit from a read result."""
    address = result.address
    kind = getattr(address.kind, "value", str(address.kind))
    text = _unit_text(
        result.content,
        result.file_path,
        address.location,
        address.summary,
        result.metadata,
        address.metadata,
    )
    content_text = _evidence_body_text(_unit_text(result.content))
    return EvidenceUnit(
        index=index,
        kind=kind,
        file_path=result.file_path,
        location=address.location,
        text=text,
        content_text=content_text,
        result=result,
        numbers=_numbers(content_text),
        alignment_score=_alignment_score(contract, text),
        roles=_initial_result_roles(result),
    )


def _address_unit(index: int, address: Address, contract: _QueryContract) -> EvidenceUnit:
    """Build a normalized evidence unit from an address candidate."""
    kind = getattr(address.kind, "value", str(address.kind))
    text = _unit_text(
        address.summary,
        str(address.metadata.get("source_path", "")),
        address.location,
        address.summary,
        address.metadata,
    )
    content_text = _unit_text(address.summary)
    return EvidenceUnit(
        index=index,
        kind=kind,
        file_path=str(address.metadata.get("source_path", "")),
        location=address.location,
        text=text,
        content_text=content_text,
        address=address,
        numbers=_numbers(content_text),
        alignment_score=_alignment_score(contract, text),
    )


def _with_compiler_metadata(
    result: ReadResult,
    unit: EvidenceUnit,
    rank: int,
    min_sources: int,
    contract: _QueryContract,
) -> ReadResult:
    """Return a copy of a read result annotated with compiler metadata."""
    metadata = dict(result.metadata)
    metadata["evidence_compiler"] = {
        "rank": rank,
        "alignment_score": unit.alignment_score,
        "roles": list(unit.roles),
        "min_sources": min_sources,
        "contract": _contract_snapshot(contract),
    }
    return ReadResult(
        address=result.address,
        content=result.content,
        file_path=result.file_path,
        line_range=result.line_range,
        metadata=metadata,
    )


def _latest_date_value(text: str) -> int | None:
    """Return a comparable date value for ISO-like dates inside text."""
    values = [
        int(year) * 10000 + int(month) * 100 + int(day)
        for year, month, day in _ISO_DATE_PATTERN.findall(text)
    ]
    return max(values) if values else None


def _initial_result_roles(result: ReadResult) -> tuple[str, ...]:
    """Return evidence-closure roles that the returned result actually proves."""
    closure = result.metadata.get("evidence_closure")
    if not isinstance(closure, dict):
        return ()
    roles: list[str] = []
    role = str(closure.get("role") or "")
    kind = getattr(result.address.kind, "value", str(result.address.kind))
    if role.startswith("required_") and kind == role.removeprefix("required_"):
        roles.append(role)
    elif role.startswith("bridge_document:"):
        document = role.removeprefix("bridge_document:")
        if _contains_term_variant(
            _unit_text(result.content, result.file_path, result.address.location),
            document,
        ):
            roles.append(role)
    for bridge in closure.get("bridges", []):
        bridge_text = str(bridge)
        if not _EXACT_IDENTIFIER_PATTERN.fullmatch(bridge_text):
            continue
        if _contains_identifier(result.content, bridge_text):
            roles.append(f"bridge:{bridge_text}")
    return tuple(dict.fromkeys(roles))


def _metadata(
    contract: _QueryContract,
    units: list[EvidenceUnit],
    ordered: list[EvidenceUnit],
    min_sources: int,
    *,
    filtered_all: bool = False,
    suppressed: list[EvidenceUnit] | None = None,
) -> dict[str, Any]:
    """Build serializable compiler trace metadata."""
    return {
        "contract": _contract_snapshot(contract),
        "input_count": len(units),
        "output_count": len(ordered),
        "min_sources": min_sources,
        "filtered_all": filtered_all,
        "selected": [
            {
                "rank": index,
                "kind": unit.kind,
                "file_path": unit.file_path,
                "location": unit.location,
                "alignment_score": unit.alignment_score,
                "roles": list(unit.roles),
            }
            for index, unit in enumerate(ordered, start=1)
        ],
        "suppressed": [
            {
                "kind": unit.kind,
                "file_path": unit.file_path,
                "location": unit.location,
                "alignment_score": unit.alignment_score,
                "roles": list(unit.roles),
                "reason": "superseded_temporal_evidence",
            }
            for unit in (suppressed or [])
        ],
    }


def _contract_snapshot(contract: _QueryContract) -> dict[str, Any]:
    """Return the Pyrrho contract and literal anchors as metadata."""
    return {
        "query_contract": contract.query_contract,
        "answerability_shape": contract.answerability_shape,
        "retrieval_modality": contract.retrieval_modality,
        "retrieval_obligation": contract.retrieval_obligation,
        "identifiers": list(contract.identifiers),
        "phrase_anchors": list(contract.phrase_anchors),
        "source_anchors": list(contract.source_anchors),
        "keyword_anchors": list(contract.keyword_anchors),
        "metric_terms": list(contract.metric_terms),
        "required_modalities": list(contract.required_modalities),
        "temporal_policy": contract.temporal_policy,
    }


def _alignment_score(contract: _QueryContract, text: str) -> int:
    """Score literal query anchor overlap in evidence text."""
    score = _hard_anchor_score(contract, text)
    normalized = _normalize_text(text)
    for term in contract.keyword_anchors:
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            score += 1
    return score


def _hard_anchor_score(contract: _QueryContract, text: str) -> int:
    """Score identifiers and phrase anchors."""
    score = 0
    for identifier in contract.identifiers:
        if _contains_identifier(text, identifier):
            score += 4
    for phrase in contract.phrase_anchors:
        if _contains_phrase(text, phrase):
            score += 3
    return score


def _identity_hard_anchor_score(contract: _QueryContract, unit: EvidenceUnit) -> int:
    """Score hard anchors only against evidence body and source identity."""
    return _hard_anchor_score(
        contract, _unit_text(unit.content_text, unit.location, unit.file_path)
    )


def _closure_bridge_content_score(unit: EvidenceUnit) -> int:
    """Score closure bridges only when the returned evidence contains the bridge."""
    if unit.result is None:
        return 0
    closure = unit.result.metadata.get("evidence_closure")
    if not isinstance(closure, dict):
        return 0
    if any(role.startswith("bridge_document:") for role in unit.roles):
        hard_anchors = [
            str(value)
            for key in ("contract_identifiers", "contract_phrase_anchors")
            for value in closure.get(key, [])
        ]
        observed_bridges = {_normalize_text(str(value)) for value in closure.get("bridges", [])}
        if any(_normalize_text(anchor) in observed_bridges for anchor in hard_anchors):
            return 4
    role = str(closure.get("role") or "")
    if not (role.startswith("required_") or role.startswith("bridge:")):
        return 0
    score = 0
    for bridge in closure.get("bridges", []):
        bridge_text = str(bridge)
        if _EXACT_IDENTIFIER_PATTERN.fullmatch(bridge_text):
            if _contains_identifier(unit.content_text, bridge_text):
                score += 4
            continue
        if _contains_term_variant(unit.content_text, bridge_text):
            score += 1
    return score


def _best_unit(contract: _QueryContract, units: list[EvidenceUnit]) -> EvidenceUnit | None:
    """Return best unit by alignment and original rank."""
    if not units:
        return None
    return sorted(units, key=lambda item: _unit_order_key(contract, item))[0]


def _best_modality_unit(
    contract: _QueryContract,
    units: list[EvidenceUnit],
) -> EvidenceUnit | None:
    """Return the required-modality unit with strongest literal anchor match."""
    if not units:
        return None
    return sorted(
        units,
        key=lambda item: (
            -_modality_match_score(contract, item),
            -item.alignment_score,
            -_temporal_preference_score(contract, item),
            item.index,
        ),
    )[0]


def _modality_match_score(contract: _QueryContract, unit: EvidenceUnit) -> int:
    """Score required modality candidates by literal query anchors."""
    score = 0
    for term in _specific_anchor_terms(contract):
        if _contains_term_variant(unit.content_text, term):
            score += 3
        elif _contains_term_variant(unit.text, term):
            score += 1
    if unit.kind == "symbol":
        score += _symbol_identity_score(contract, unit)
    return score


def _content_alignment_score(contract: _QueryContract, unit: EvidenceUnit) -> int:
    """Score only the evidence body, excluding path and aggregate metadata."""
    score = _hard_anchor_score(contract, unit.content_text)
    for term in contract.keyword_anchors:
        if _contains_term_variant(unit.content_text, term):
            score += 1
    return score


def _unit_order_key(contract: _QueryContract, unit: EvidenceUnit) -> tuple[int, int, int, int]:
    """Return the stable compiler ordering key for an evidence unit."""
    return (
        -_content_alignment_score(contract, unit),
        -unit.alignment_score,
        -_temporal_preference_score(contract, unit),
        unit.index,
    )


def _temporal_preference_score(contract: _QueryContract, unit: EvidenceUnit) -> int:
    """Prefer final/current/latest spans only when Pyrrho marked temporal grounding."""
    if contract.temporal_policy != "temporal":
        return 0
    query_terms = set(_normalize_text(contract.query).split())
    text = _normalize_text(_unit_text(unit.content_text, unit.location))
    score = 0
    if "final" in query_terms and ("final" in text or "confirmed" in text):
        score += 100_000_000
    if query_terms.intersection({"latest", "current"}):
        if "current" in text:
            score += 100_000_000
        date_value = _latest_date_value(unit.content_text)
        if date_value is not None:
            score += date_value
    return score


def _symbol_identity_score(contract: _QueryContract, unit: EvidenceUnit) -> int:
    """Prefer code symbols whose metadata identity matches literal query terms."""
    metadata = {}
    if unit.result is not None:
        metadata.update(unit.result.metadata)
        metadata.update(unit.result.address.metadata)
    elif unit.address is not None:
        metadata.update(unit.address.metadata)

    name = str(metadata.get("name", ""))
    qualified_name = str(metadata.get("qualified_name", unit.location))
    identity = _unit_text(name, qualified_name, unit.location)
    score = 0
    for identifier in contract.identifiers:
        if _contains_identifier(identity, identifier):
            score += 8
    for term in _specific_anchor_terms(contract):
        if _contains_term_variant(identity, term):
            score += 3
    return score


def _specific_anchor_terms(contract: _QueryContract) -> tuple[str, ...]:
    """Return query terms that distinguish one candidate from another."""
    generic = {
        "applies",
        "confirm",
        "enabled",
        "many",
        "mentioned",
        "release",
        "rollout",
    }
    terms = [term for term in contract.keyword_anchors if term not in generic]
    return tuple(dict.fromkeys(terms))


def _needs_candidate_rescue(contract: _QueryContract) -> bool:
    """Return whether pre-read address rescue should run."""
    return bool(contract.required_modalities)


def _role_is_contract_obligation(role: str) -> bool:
    """Return whether closure/compiler role should be ordered before residual evidence."""
    return (
        role.startswith("required_")
        or role.startswith("anchor_identifier:")
        or role.startswith("anchor_phrase:")
        or role.startswith("bridge:")
        or role.startswith("bridge_document:")
    )


def _bridge_companion_units(
    contract: _QueryContract,
    units: list[EvidenceUnit],
    selected: list[EvidenceUnit],
) -> list[tuple[EvidenceUnit, str]]:
    """Return exact bridge companions from the full read set."""
    if not _contract_allows_bridge_companions(contract):
        return []
    selected_keys = {(unit.kind, unit.file_path, unit.location) for unit in selected}
    companions: list[tuple[EvidenceUnit, str]] = []
    companion_keys: set[tuple[str, str, str]] = set()

    for term in _bridge_companion_terms(contract, selected):
        matches = [
            unit
            for unit in units
            if (unit.kind, unit.file_path, unit.location) not in selected_keys
            and (unit.kind, unit.file_path, unit.location) not in companion_keys
            and _bridge_term_matches_unit(term, unit)
        ]
        if not matches:
            continue

        by_kind: dict[str, list[EvidenceUnit]] = {}
        for match in matches:
            by_kind.setdefault(match.kind, []).append(match)

        for kind in ("table", "symbol", "section"):
            kind_matches = by_kind.get(kind)
            if not kind_matches:
                continue
            match = _best_bridge_companion(contract, term, kind_matches)
            key = (match.kind, match.file_path, match.location)
            companion_keys.add(key)
            companions.append((match, term))
            if len(companions) >= 6:
                return companions
    return companions


def _contract_allows_bridge_companions(contract: _QueryContract) -> bool:
    """Return whether Pyrrho requested cross-source companion coverage."""
    return contract.retrieval_modality == "mixed" or len(contract.required_modalities) > 1


def _bridge_companion_terms(
    contract: _QueryContract,
    selected: list[EvidenceUnit],
) -> tuple[str, ...]:
    """Extract bridge terms worth using for companion-source selection."""
    terms: list[str] = []
    for unit in selected:
        for role in unit.roles:
            if role.startswith("bridge:"):
                terms.append(role.removeprefix("bridge:"))
        text = _unit_text(unit.content_text, unit.location, unit.file_path)
        terms.extend(
            match.group(0).strip(".,;:()[]{}") for match in _EXACT_IDENTIFIER_PATTERN.finditer(text)
        )
        terms.extend(_table_hint_terms(text))

    query_terms = set(_normalize_text(contract.query).split())
    deduped: list[str] = []
    seen: set[str] = set()
    for term in terms:
        normalized = _normalize_text(term)
        if not normalized or normalized in seen:
            continue
        if normalized in query_terms and not _EXACT_IDENTIFIER_PATTERN.fullmatch(term):
            continue
        if not _is_bridge_companion_term(term):
            continue
        seen.add(normalized)
        deduped.append(term)
    return tuple(deduped)


def _table_hint_terms(text: str) -> tuple[str, ...]:
    """Return table-name hints such as 'alerts table' from bridge prose."""
    hints: list[str] = []
    for match in re.finditer(r"\b([A-Za-z][A-Za-z0-9_-]{2,})\s+table\b", text, re.IGNORECASE):
        hints.append(f"{match.group(1)} table")
    return tuple(hints)


def _is_bridge_companion_term(term: str) -> bool:
    """Return whether a bridge token is specific enough to pull a companion."""
    value = term.strip()
    normalized = _normalize_text(value)
    if len(normalized) < 3:
        return False
    if _EXACT_IDENTIFIER_PATTERN.fullmatch(value):
        return True
    return bool(re.search(r"[\d_-]", value) or normalized.endswith("table"))


def _bridge_term_matches_unit(term: str, unit: EvidenceUnit) -> bool:
    """Return whether a unit contains a bridge term in source identity or body."""
    text = _unit_text(unit.content_text, unit.location, unit.file_path)
    if _EXACT_IDENTIFIER_PATTERN.fullmatch(term):
        return _contains_identifier(text, term)
    return _contains_phrase(text, term) or _contains_term_variant(text, term)


def _best_bridge_companion(
    contract: _QueryContract,
    term: str,
    units: list[EvidenceUnit],
) -> EvidenceUnit:
    """Return the strongest companion for a bridge term within one modality."""
    return sorted(
        units,
        key=lambda item: (
            -_bridge_companion_score(term, item),
            -item.alignment_score,
            -_content_alignment_score(contract, item),
            item.index,
        ),
    )[0]


def _bridge_companion_score(term: str, unit: EvidenceUnit) -> int:
    """Score exact bridge evidence without relying on rank alone."""
    identity = _unit_text(unit.location, unit.file_path)
    body = unit.content_text
    score = 0
    if _EXACT_IDENTIFIER_PATTERN.fullmatch(term):
        if _contains_identifier(body, term):
            score += 8
        if _contains_identifier(identity, term):
            score += 4
    else:
        if _contains_phrase(body, term) or _contains_term_variant(body, term):
            score += 4
        if _contains_phrase(identity, term) or _contains_term_variant(identity, term):
            score += 2
    if unit.kind in {"table", "symbol"}:
        score += 2
    return score


def _role_requires_prefix_floor(role: str, kind: str | None = None) -> bool:
    """Return whether a compiler role should force Pyrrho to wait for this source."""
    if role.startswith("required_"):
        modality = role.removeprefix("required_")
        return kind is None or kind == modality
    return (
        role.startswith("anchor_identifier:")
        or role.startswith("anchor_phrase:")
        or role.startswith("source_anchor:")
        or role.startswith("bridge:")
        or role in {"conflict_value", "latest", "final"}
    )


def _with_role(unit: EvidenceUnit, role: str) -> EvidenceUnit:
    """Return a unit copy with an additional compiler role."""
    roles = tuple(dict.fromkeys((*unit.roles, role)))
    return EvidenceUnit(
        index=unit.index,
        kind=unit.kind,
        file_path=unit.file_path,
        location=unit.location,
        text=unit.text,
        content_text=unit.content_text,
        result=unit.result,
        address=unit.address,
        numbers=unit.numbers,
        alignment_score=unit.alignment_score,
        roles=roles,
    )


def _numbers(text: str) -> tuple[str, ...]:
    """Extract numeric facts as observed evidence metadata."""
    return tuple(dict.fromkeys(match.group(0).lower() for match in _NUMBER_PATTERN.finditer(text)))


def _unit_text(*parts: Any) -> str:
    """Join evidence fields into one searchable string."""
    flattened: list[str] = []
    for part in parts:
        if isinstance(part, dict):
            flattened.extend(str(value) for value in part.values() if isinstance(value, str))
            continue
        if isinstance(part, (list, tuple, set)):
            flattened.extend(str(value) for value in part)
            continue
        if part is not None:
            flattened.append(str(part))
    return "\n".join(flattened)


def _evidence_body_text(text: str) -> str:
    """Return factual body text, excluding parser comments and child TOCs."""
    kept: list[str] = []
    in_child_toc = False
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            in_child_toc = False
            kept.append(line)
            continue
        if stripped.startswith("<!--") and stripped.endswith("-->"):
            continue
        if stripped == "Subsections:":
            in_child_toc = True
            continue
        if in_child_toc and stripped.startswith("-"):
            continue
        kept.append(line)
    return "\n".join(kept).strip()


def _contains_identifier(text: str, identifier: str) -> bool:
    """Return whether raw text contains the literal identifier."""
    return _contains_exact_identifier(text, identifier)


def _contains_phrase(text: str, phrase: str) -> bool:
    """Return whether normalized text contains all phrase terms."""
    terms = _normalize_text(phrase).split()
    normalized = _normalize_text(text)
    return all(_contains_normalized_term(normalized, term) for term in terms)


def _contains_term_variant(text: str, term: str) -> bool:
    """Return whether text contains a term or simple singular/plural variant."""
    return _contains_normalized_term(_normalize_text(text), _normalize_text(term))


def _contains_normalized_term(normalized_text: str, normalized_term: str) -> bool:
    """Return whether normalized text contains a term with basic number variants."""
    if not normalized_term:
        return False
    variants = {normalized_term}
    if normalized_term.endswith("s") and len(normalized_term) > 3:
        variants.add(normalized_term[:-1])
    else:
        variants.add(f"{normalized_term}s")
    return any(re.search(rf"\b{re.escape(variant)}\b", normalized_text) for variant in variants)


__all__ = [
    "EvidenceCompilation",
    "EvidenceUnit",
    "compile_evidence",
    "order_addresses_for_contract",
    "query_has_code_obligation",
    "query_has_table_obligation",
]

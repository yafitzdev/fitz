# fitz_sage/engines/fitz_krag/evidence_compiler.py
"""Pyrrho-contract evidence compilation for KRAG evidence packs."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from fitz_sage.engines.fitz_krag.types import Address, ReadResult
from fitz_sage.governance.evidence_contract import (
    EXACT_IDENTIFIER_PATTERN as _EXACT_IDENTIFIER_PATTERN,
)
from fitz_sage.governance.evidence_contract import QueryContract as _QueryContract
from fitz_sage.governance.evidence_contract import build_query_contract as _build_query_contract
from fitz_sage.governance.evidence_contract import normalize_text as _normalize_text

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
    """Compile ranked read results into a Pyrrho-contract evidence order."""
    contract = _build_query_contract(query, profile)
    if not results:
        return EvidenceCompilation([], _metadata(contract, [], [], 0))

    units = [_read_result_unit(index, result, contract) for index, result in enumerate(results)]
    aligned = _aligned_units(contract, units)
    if contract.has_hard_anchors and not aligned:
        return EvidenceCompilation([], _metadata(contract, units, [], 0, filtered_all=True))

    working = aligned if aligned else units
    ordered = _compile_order(contract, working)
    min_sources = _minimum_sources(contract, ordered)
    compiled = [
        _with_compiler_metadata(unit.result, unit, rank, min_sources, contract)
        for rank, unit in enumerate(ordered, start=1)
        if unit.result is not None
    ]
    return EvidenceCompilation(compiled, _metadata(contract, units, ordered, min_sources))


def order_addresses_for_contract(
    query: str,
    candidates: list[Address],
    selected: list[Address],
    profile: Any = None,
) -> list[Address]:
    """Preserve Pyrrho-required address kinds before the read step."""
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
            unit
            for unit in candidate_units
            if unit.kind == modality
            and unit.address is not None
            and (unit.address.source_id, unit.address.location) not in selected_keys
        ]
        if not matches:
            continue
        match = _best_modality_unit(contract, matches)
        if match is None or match.address is None:
            continue
        selected_keys.add((match.address.source_id, match.address.location))
        rescued.append(match.address)

    if not rescued:
        return selected
    return selected + rescued


def query_has_table_obligation(query: str, profile: Any = None) -> bool:
    """Return whether Pyrrho requires table evidence."""
    contract = _build_query_contract(query, profile)
    return "table" in contract.required_modalities


def query_has_code_obligation(query: str, profile: Any = None) -> bool:
    """Return whether Pyrrho requires code-symbol evidence."""
    contract = _build_query_contract(query, profile)
    return "symbol" in contract.required_modalities


def _compile_order(contract: _QueryContract, units: list[EvidenceUnit]) -> list[EvidenceUnit]:
    """Build evidence order from Pyrrho obligations and literal alignment."""
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

    for phrase in contract.phrase_anchors:
        phrase_units = [unit for unit in units if _contains_phrase(unit.text, phrase)]
        match = _best_unit(contract, phrase_units)
        if match is not None:
            add(match, f"anchor:{phrase}")

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
    obligation_roles = {
        role for unit in units for role in unit.roles if _role_is_contract_obligation(role)
    }
    required = max(required, len(obligation_roles))
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
    result: ReadResult | None,
    unit: EvidenceUnit,
    rank: int,
    min_sources: int,
    contract: _QueryContract,
) -> ReadResult | None:
    """Return a copy of a read result annotated with compiler metadata."""
    if result is None:
        return None
    content, span_metadata = _focused_content(result.content, contract)
    metadata = dict(result.metadata)
    metadata["evidence_compiler"] = {
        "rank": rank,
        "alignment_score": unit.alignment_score,
        "roles": list(unit.roles),
        "min_sources": min_sources,
        "contract": _contract_snapshot(contract),
    }
    if span_metadata:
        metadata["evidence_span"] = span_metadata
    return ReadResult(
        address=result.address,
        content=content,
        file_path=result.file_path,
        line_range=result.line_range,
        metadata=metadata,
    )


def _focused_content(content: str, contract: _QueryContract) -> tuple[str, dict[str, Any] | None]:
    """Return a narrower evidence span when one addressed unit contains separable facts."""
    prefix, body = _content_prefix(content)
    blocks = _paragraph_blocks(body)
    if len(blocks) < 2:
        return content, None

    scored = [
        (_span_score(block, contract, index, blocks), index, block)
        for index, block in enumerate(blocks)
    ]
    best_score, best_index, best_block = sorted(scored, key=lambda item: (-item[0], item[1]))[0]
    if best_score <= 0:
        return content, None

    focused = "\n\n".join(part for part in (prefix, best_block.strip()) if part)
    if focused.strip() == content.strip():
        return content, None
    return focused, {
        "kind": "paragraph",
        "selected_index": best_index + 1,
        "block_count": len(blocks),
        "score": best_score,
        "original_char_count": len(content),
        "focused_char_count": len(focused),
    }


def _content_prefix(content: str) -> tuple[str, str]:
    """Split a display prefix such as a breadcrumb from the body text."""
    lines = content.splitlines()
    if not lines:
        return "", content
    first = lines[0].strip()
    if re.fullmatch(r"\[[^\]]+\]", first) and len(lines) > 1:
        return first, "\n".join(lines[1:]).lstrip()
    return "", content


def _paragraph_blocks(content: str) -> list[str]:
    """Split content into non-empty paragraph blocks."""
    return [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]


def _span_score(
    block: str,
    contract: _QueryContract,
    index: int,
    blocks: list[str],
) -> int:
    """Score a paragraph span by literal alignment and temporal/finality cues."""
    score = _hard_anchor_score(contract, block) * 10
    for term in contract.keyword_anchors:
        if _contains_term_variant(block, term):
            score += 1

    query_terms = set(_normalize_text(contract.query).split())
    normalized = _normalize_text(block)
    if "final" in query_terms and (
        "final" in normalized or "confirmed" in normalized
    ):
        score += 20
    if query_terms.intersection({"latest", "current"}):
        if "current" in normalized:
            score += 20
        score += _date_rank_bonus(block, blocks)
    return score


def _date_rank_bonus(block: str, blocks: list[str]) -> int:
    """Return a small deterministic bonus for the latest dated paragraph."""
    block_date = _latest_date_value(block)
    if block_date is None:
        return 0
    dated_values = sorted(
        {value for candidate in blocks if (value := _latest_date_value(candidate)) is not None}
    )
    if not dated_values:
        return 0
    return dated_values.index(block_date) + 1


def _latest_date_value(text: str) -> int | None:
    """Return a comparable date value for ISO-like dates inside text."""
    values = [
        int(year) * 10000 + int(month) * 100 + int(day)
        for year, month, day in _ISO_DATE_PATTERN.findall(text)
    ]
    return max(values) if values else None


def _initial_result_roles(result: ReadResult) -> tuple[str, ...]:
    """Return structural roles already proven by evidence closure."""
    closure = result.metadata.get("evidence_closure")
    if not isinstance(closure, dict):
        return ()
    roles: list[str] = []
    role = closure.get("role")
    if isinstance(role, str) and role:
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
    }


def _contract_snapshot(contract: _QueryContract) -> dict[str, Any]:
    """Return the Pyrrho contract and literal anchors as metadata."""
    return {
        "query_contract": contract.query_contract,
        "route": contract.route,
        "answerability_shape": contract.answerability_shape,
        "retrieval_modality": contract.retrieval_modality,
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
    return _hard_anchor_score(contract, _unit_text(unit.content_text, unit.location, unit.file_path))


def _closure_bridge_content_score(unit: EvidenceUnit) -> int:
    """Score closure bridges only when the returned evidence contains the bridge."""
    if unit.result is None:
        return 0
    closure = unit.result.metadata.get("evidence_closure")
    if not isinstance(closure, dict):
        return 0
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
    """Return whether closure/compiler role contributes to a Pyrrho prefix floor."""
    return (
        role.startswith("required_")
        or role.startswith("bridge:")
        or role.startswith("bridge_document:")
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
    """Return whether raw text contains an identifier or separator variant."""
    lower = text.lower()
    normalized = identifier.lower()
    variants = {
        normalized,
        normalized.replace("_", "-"),
        normalized.replace("-", "_"),
        normalized.replace("_", " ").replace("-", " "),
    }
    for variant in variants:
        escaped = re.escape(variant).replace(r"\ ", r"\s+")
        if re.search(rf"(?<![A-Za-z0-9_]){escaped}(?![A-Za-z0-9_])", lower):
            return True
    return False


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

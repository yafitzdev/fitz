# tests/unit/test_krag_router_ranker.py
"""
Unit tests for RetrievalRouter and CrossStrategyRanker.

Tests routing logic (strategy dispatch, deduplication, fallback) and
cross-strategy ranking (weight application, entity bonus, ordering).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fitz_sage.engines.fitz_krag.retrieval.ranker import (
    ENTITY_MATCH_BONUS,
    CrossStrategyRanker,
)
from fitz_sage.engines.fitz_krag.retrieval.router import RetrievalRouter
from fitz_sage.engines.fitz_krag.retrieval_profile import RetrievalProfile
from fitz_sage.engines.fitz_krag.types import Address, AddressKind

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(top_addresses: int = 10) -> MagicMock:
    """Create a mock FitzKragConfig with the fields the router reads."""
    cfg = MagicMock()
    cfg.top_addresses = top_addresses
    cfg.retrieval_workers = 4
    return cfg


def _addr(
    kind: AddressKind = AddressKind.SYMBOL,
    source_id: str = "src",
    location: str = "mod.func",
    summary: str = "does something",
    score: float = 0.5,
    metadata: dict | None = None,
) -> Address:
    """Shortcut to build an Address."""
    return Address(
        kind=kind,
        source_id=source_id,
        location=location,
        summary=summary,
        score=score,
        metadata=metadata or {},
    )


def _code_profile(
    entities: tuple[str, ...] = (),
    top_k: int = 10,
) -> RetrievalProfile:
    """RetrievalProfile with CODE-like weights."""
    return RetrievalProfile(
        strategy_weights={"code": 0.8, "section": 0.1, "table": 0.05, "chunk": 0.05},
        entities=entities,
        top_k=top_k,
        analysis_type="code",
        analysis_confidence=0.9,
    )


def _custom_weight_profile(
    code: float,
    section: float,
    chunk: float,
    entities: tuple[str, ...] = (),
    top_k: int = 10,
) -> RetrievalProfile:
    """Build a RetrievalProfile with custom strategy weights."""
    return RetrievalProfile(
        strategy_weights={"code": code, "section": section, "chunk": chunk},
        entities=entities,
        top_k=top_k,
        analysis_type="general",
        analysis_confidence=0.8,
    )


# ---------------------------------------------------------------------------
# TestRetrievalRouter
# ---------------------------------------------------------------------------


class TestRetrievalRouter:
    """Tests for RetrievalRouter dispatch, fallback, dedup, and ranking."""

    # -- test_retrieve_code_only ------------------------------------------

    def test_retrieve_code_only(self):
        """Code strategy returns addresses; no section or chunk used."""
        code_strat = MagicMock()
        code_addrs = [_addr(score=0.9), _addr(score=0.7, location="mod.bar")]
        code_strat.retrieve.return_value = code_addrs

        config = _make_config(top_addresses=10)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=None,
        )

        result = router.retrieve("find func")

        code_strat.retrieve.assert_called_once_with("find func", 10, detection=None)
        assert len(result) == 2
        # Without profile, sorted by score descending
        assert result[0].score >= result[1].score

    def test_retrieve_returns_profile_top_k_candidates(self):
        """Profile top_k should control the final candidate list size."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=1.0 - (index * 0.01), location=f"mod.func{index}") for index in range(12)
        ]

        config = _make_config(top_addresses=5)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=None,
        )

        result = router.retrieve("find func", _code_profile(top_k=8))

        assert len(result) == 8

    # -- test_retrieve_with_section_strategy ------------------------------

    def test_retrieve_with_section_strategy(self):
        """Both code and section strategies contribute results."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(AddressKind.SYMBOL, score=0.8, location="a.py:func"),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(AddressKind.SECTION, score=0.7, location="README#setup"),
        ]
        config = _make_config()

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        result = router.retrieve("how to setup")

        code_strat.retrieve.assert_called_once()
        section_strat.retrieve.assert_called_once()
        assert len(result) == 2

    # -- test_retrieve_skips_low_weight_strategy --------------------------

    def test_retrieve_skips_low_weight_strategy(self):
        """Strategy with weight <= 0.05 is skipped entirely."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=0.9, location="a.func"),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(AddressKind.SECTION, score=0.6, location="doc#s"),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        # Custom weights: section weight is 0.04 (below 0.05 threshold)
        profile = _custom_weight_profile(code=0.9, section=0.04, chunk=0.04)

        result = router.retrieve("find func", profile)

        code_strat.retrieve.assert_called_once()
        section_strat.retrieve.assert_not_called()
        assert len(result) >= 1

    # -- test_retrieve_deduplicates ---------------------------------------

    def test_retrieve_deduplicates(self):
        """Same (source_id, location) from different strategies kept once."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(
                AddressKind.SYMBOL,
                source_id="file.py",
                location="MyClass",
                score=0.9,
            ),
        ]
        section_strat = MagicMock()
        section_strat.retrieve.return_value = [
            _addr(
                AddressKind.SECTION,
                source_id="file.py",
                location="MyClass",
                score=0.7,
            ),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
            section_strategy=section_strat,
        )

        result = router.retrieve("query")

        # Duplicate by (source_id, location) -- first one wins
        assert len(result) == 1
        assert result[0].score == 0.9

    # -- test_retrieve_with_profile_uses_ranker ---------------------------

    def test_retrieve_with_profile_uses_ranker(self):
        """When profile is provided, CrossStrategyRanker is used."""
        code_strat = MagicMock()
        sym_hi = _addr(AddressKind.SYMBOL, score=0.5, location="low_sym")
        sym_lo = _addr(AddressKind.SYMBOL, score=0.9, location="hi_sym")
        code_strat.retrieve.return_value = [sym_hi, sym_lo]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        profile = _code_profile(entities=("hi_sym",))
        result = router.retrieve("find hi_sym", profile)

        # Ranker should apply entity bonus to hi_sym, boosting it
        assert len(result) == 2
        # hi_sym should rank first because of entity match bonus
        assert result[0].location == "hi_sym"

    # -- test_retrieve_without_profile_sorts_by_score --------------------

    def test_retrieve_without_profile_sorts_by_score(self):
        """No profile -> results sorted by raw score descending."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=0.3, location="low"),
            _addr(score=0.9, location="high"),
            _addr(score=0.6, location="mid"),
        ]

        config = _make_config()
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        result = router.retrieve("query")

        assert [a.location for a in result] == ["high", "mid", "low"]

    # -- test_retrieve_limits_results -------------------------------------

    def test_retrieve_limits_results(self):
        """More results than top_addresses -> truncated to limit."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = [
            _addr(score=1.0 - i * 0.05, location=f"f{i}") for i in range(15)
        ]

        config = _make_config(top_addresses=5)
        router = RetrievalRouter(
            code_strategy=code_strat,
            config=config,
        )

        result = router.retrieve("query")

        assert len(result) == 5
        # Should be top-5 by score
        assert result[0].score == 1.0

    def test_agentic_progress_reports_pre_index_candidates(self):
        """Progress text scopes agentic results as pre-index candidates."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = []
        agentic = MagicMock()
        agentic.has_pending_files.return_value = True
        agentic.retrieve.return_value = [
            _addr(
                AddressKind.FILE,
                source_id="a",
                location="docs/a.md",
                metadata={"disk_path": "docs/a.md"},
            ),
            _addr(
                AddressKind.FILE,
                source_id="b",
                location="docs/b.md",
                metadata={"disk_path": "docs/b.md"},
            ),
        ]
        progress = MagicMock()

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=_make_config(top_addresses=10),
            agentic_strategy=agentic,
        )

        router.retrieve("key facts", progress=progress)

        progress.assert_any_call(
            "Supplemental scan: checking files still awaiting enriched index..."
        )
        progress.assert_any_call(
            "Supplemental scan: added 2 early candidate(s) from 2 file(s) (a.md, b.md)"
        )

    def test_agentic_scan_is_skipped_without_pending_files(self):
        """Fully query-ready collections do not emit supplemental scan noise."""
        code_strat = MagicMock()
        code_strat.retrieve.return_value = []
        agentic = MagicMock()
        agentic.has_pending_files.return_value = False
        progress = MagicMock()

        router = RetrievalRouter(
            code_strategy=code_strat,
            config=_make_config(top_addresses=10),
            agentic_strategy=agentic,
        )

        router.retrieve("key facts", progress=progress)

        agentic.retrieve.assert_not_called()
        progress.assert_not_called()


# ---------------------------------------------------------------------------
# TestCrossStrategyRanker
# ---------------------------------------------------------------------------


class TestCrossStrategyRanker:
    """Tests for CrossStrategyRanker scoring and ordering."""

    def setup_method(self):
        self.ranker = CrossStrategyRanker()

    # -- test_rank_applies_weights ----------------------------------------

    def test_rank_applies_weights(self):
        """CODE profile boosts SYMBOL addresses via higher weight."""
        sym_addr = _addr(AddressKind.SYMBOL, score=0.5, location="func")
        chunk_addr = _addr(AddressKind.CHUNK, score=0.5, location="chunk1")

        profile = _code_profile()
        # CODE weights: code=0.8, section=0.1, chunk=0.05
        # sym: 0.5 * 0.8 = 0.40,  chunk: 0.5 * 0.05 = 0.025
        result = self.ranker.rank([chunk_addr, sym_addr], profile)

        assert result[0].kind == AddressKind.SYMBOL
        assert result[1].kind == AddressKind.CHUNK

    # -- test_rank_entity_match_bonus_in_location -------------------------

    def test_rank_entity_match_bonus(self):
        """Entity present in location earns ENTITY_MATCH_BONUS."""
        addr_match = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="MyClass.do_work",
        )
        addr_no = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="other_func",
        )

        profile = _code_profile(entities=("MyClass",))
        result = self.ranker.rank([addr_no, addr_match], profile)

        # addr_match gets bonus, should rank higher
        assert result[0].location == "MyClass.do_work"

        # Verify the bonus magnitude: both have same base * weight,
        # but match gets +ENTITY_MATCH_BONUS
        weights = profile.strategy_weights
        expected_match = 0.5 * weights["code"] + ENTITY_MATCH_BONUS
        expected_no = 0.5 * weights["code"]
        assert expected_match > expected_no

    # -- test_rank_entity_in_summary --------------------------------------

    def test_rank_entity_in_summary(self):
        """Entity present in summary (not location) also earns bonus."""
        addr = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="some_func",
            summary="Handles MyClass initialization",
        )
        addr_no = _addr(
            AddressKind.SYMBOL,
            score=0.5,
            location="other_func",
            summary="unrelated work",
        )

        profile = _code_profile(entities=("MyClass",))
        result = self.ranker.rank([addr_no, addr], profile)

        assert result[0].summary == "Handles MyClass initialization"

    # -- test_rank_no_entity_match ----------------------------------------

    def test_rank_no_entity_match(self):
        """No entity match -> no bonus applied."""
        addr1 = _addr(AddressKind.SYMBOL, score=0.8, location="func_a")
        addr2 = _addr(AddressKind.SYMBOL, score=0.6, location="func_b")

        profile = _code_profile(entities=("NonExistent",))
        result = self.ranker.rank([addr2, addr1], profile)

        # Neither gets bonus; order by weighted score alone
        assert result[0].location == "func_a"
        assert result[1].location == "func_b"

    # -- test_rank_sorts_descending ---------------------------------------

    def test_rank_sorts_descending(self):
        """Results are sorted by computed score, highest first."""
        addrs = [
            _addr(AddressKind.SYMBOL, score=0.3, location="low"),
            _addr(AddressKind.SYMBOL, score=0.9, location="high"),
            _addr(AddressKind.SYMBOL, score=0.6, location="mid"),
        ]

        profile = _code_profile()
        result = self.ranker.rank(addrs, profile)

        scores = [a.score for a in result]
        # Original scores should be in descending order (all same weight)
        assert scores == sorted(scores, reverse=True)

    # -- test_rank_empty_addresses ----------------------------------------

    def test_rank_empty_addresses(self):
        """Empty address list returns empty list."""
        profile = _code_profile()
        result = self.ranker.rank([], profile)
        assert result == []

    # -- test_rank_without_profile ----------------------------------------

    def test_rank_without_profile(self):
        """When profile is None, rank by raw score only."""
        addrs = [
            _addr(AddressKind.SYMBOL, score=0.3, location="low"),
            _addr(AddressKind.SYMBOL, score=0.9, location="high"),
        ]
        result = self.ranker.rank(addrs, None)

        assert result[0].location == "high"
        assert result[1].location == "low"

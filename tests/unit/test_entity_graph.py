# tests/unit/test_entity_graph.py
"""Tests for entity graph store and related chunk discovery."""

from __future__ import annotations

import uuid

import pytest

from fitz_sage.retrieval.entity_graph import EntityGraphStore


class TestEntityGraphStore:
    """Test EntityGraphStore functionality."""

    @pytest.fixture
    def store(self):
        """Create a test entity graph store with unique collection."""
        # Use unique collection name to avoid test interference
        collection = f"test_entity_graph_{uuid.uuid4().hex[:8]}"
        store = EntityGraphStore(collection=collection)
        store.clear()  # Ensure clean state
        yield store
        store.clear()  # Clean up after test
        store.close()

    def test_add_and_retrieve_entities(self, store):
        """Test adding entities and retrieving them."""
        # Add entities for chunk1
        store.add_chunk_entities(
            "chunk1",
            [
                ("Sarah Chen", "person"),
                ("Acme Corp", "company"),
            ],
        )

        # Add entities for chunk2 (shares "Acme Corp")
        store.add_chunk_entities(
            "chunk2",
            [
                ("Acme Corp", "company"),
                ("Widget Factory", "location"),
            ],
        )

        # Add entities for chunk3 (shares "Sarah Chen")
        store.add_chunk_entities(
            "chunk3",
            [
                ("Sarah Chen", "person"),
                ("Board Meeting", "event"),
            ],
        )

        # Verify stats
        stats = store.stats()
        # Unique entities: Sarah Chen, Acme Corp, Widget Factory, Board Meeting
        assert stats["entities"] == 4
        assert stats["edges"] == 6  # 6 entity-chunk connections

    def test_get_related_chunks_via_shared_entities(self, store):
        """Test finding related chunks via shared entities."""
        # Setup: 3 chunks with overlapping entities
        store.add_chunk_entities(
            "chunk1",
            [("Sarah Chen", "person"), ("Acme Corp", "company")],
        )
        store.add_chunk_entities(
            "chunk2",
            [("Acme Corp", "company"), ("Q4 Report", "document")],
        )
        store.add_chunk_entities(
            "chunk3",
            [("Sarah Chen", "person"), ("Board Meeting", "event")],
        )
        store.add_chunk_entities(
            "chunk4",
            [("Unrelated Entity", "other")],
        )

        # Query: Given chunk1, find related chunks
        related = store.get_related_chunks(["chunk1"], max_total=10)

        # chunk2 shares "Acme Corp", chunk3 shares "Sarah Chen"
        # chunk4 doesn't share any entities
        assert "chunk2" in related
        assert "chunk3" in related
        assert "chunk4" not in related
        assert "chunk1" not in related  # Input chunk excluded

    def test_get_related_chunks_ranking(self, store):
        """Test that related chunks are ranked by shared entity count."""
        # Setup: chunk2 shares 2 entities with chunk1, chunk3 shares 1
        store.add_chunk_entities(
            "chunk1",
            [("Entity A", "type"), ("Entity B", "type"), ("Entity C", "type")],
        )
        store.add_chunk_entities(
            "chunk2",
            [("Entity A", "type"), ("Entity B", "type")],  # Shares 2
        )
        store.add_chunk_entities(
            "chunk3",
            [("Entity A", "type")],  # Shares 1
        )

        related = store.get_related_chunks(["chunk1"], max_total=10)

        # chunk2 should come first (more shared entities)
        assert related[0] == "chunk2"
        assert related[1] == "chunk3"

    def test_get_chunks_for_entity(self, store):
        """Test direct entity lookup."""
        store.add_chunk_entities("chunk1", [("Python", "language")])
        store.add_chunk_entities("chunk2", [("Python", "language")])
        store.add_chunk_entities("chunk3", [("JavaScript", "language")])

        python_chunks = store.get_chunks_for_entity("Python")
        assert set(python_chunks) == {"chunk1", "chunk2"}

    def test_get_chunks_for_multiple_entities(self, store):
        """Test lookup across multiple entities."""
        store.add_chunk_entities("chunk1", [("Python", "lang"), ("Django", "framework")])
        store.add_chunk_entities("chunk2", [("Python", "lang")])
        store.add_chunk_entities("chunk3", [("Django", "framework")])
        store.add_chunk_entities("chunk4", [("React", "framework")])

        # Chunks matching Python OR Django, ranked by match count
        chunks = store.get_chunks_for_entities(["Python", "Django"])

        # chunk1 matches both, should be first
        assert chunks[0] == "chunk1"
        assert set(chunks) == {"chunk1", "chunk2", "chunk3"}

    def test_entity_normalization(self, store):
        """Test that entity names are normalized for matching."""
        store.add_chunk_entities("chunk1", [("Sarah Chen", "person")])
        store.add_chunk_entities("chunk2", [("sarah chen", "person")])  # lowercase
        store.add_chunk_entities("chunk3", [("SARAH CHEN", "person")])  # uppercase

        # All should be treated as same entity
        stats = store.stats()
        assert stats["entities"] == 1  # Single normalized entity

        # All chunks should be related
        related = store.get_related_chunks(["chunk1"])
        assert set(related) == {"chunk2", "chunk3"}

    def test_remove_chunk(self, store):
        """Test removing a chunk's entity associations."""
        store.add_chunk_entities("chunk1", [("Entity A", "type")])
        store.add_chunk_entities("chunk2", [("Entity A", "type")])

        # Initially both chunks are related
        related = store.get_related_chunks(["chunk1"])
        assert "chunk2" in related

        # Remove chunk2
        store.remove_chunk("chunk2")

        # Now chunk1 has no related chunks
        related = store.get_related_chunks(["chunk1"])
        assert "chunk2" not in related

    def test_min_shared_entities_filter(self, store):
        """Test filtering by minimum shared entities."""
        store.add_chunk_entities(
            "chunk1",
            [("A", "t"), ("B", "t"), ("C", "t")],
        )
        store.add_chunk_entities(
            "chunk2",
            [("A", "t"), ("B", "t")],  # Shares 2
        )
        store.add_chunk_entities(
            "chunk3",
            [("A", "t")],  # Shares 1
        )

        # With min_shared_entities=2, only chunk2 qualifies
        related = store.get_related_chunks(["chunk1"], max_total=10, min_shared_entities=2)
        assert related == ["chunk2"]

    def test_empty_inputs(self, store):
        """Test handling of empty inputs."""
        # Empty chunk_ids
        assert store.get_related_chunks([]) == []

        # No entities for chunk
        store.add_chunk_entities("chunk1", [])
        assert store.get_related_chunks(["chunk1"]) == []

    def test_clear(self, store):
        """Test clearing all data."""
        store.add_chunk_entities("chunk1", [("Entity", "type")])
        assert store.stats()["entities"] == 1

        store.clear()
        assert store.stats()["entities"] == 0
        assert store.stats()["edges"] == 0


class TestEntityGraphIntegration:
    """Integration tests for entity graph with retrieval."""

    def test_entity_bridge_scenario(self):
        """
        Test entity-linked bridge expansion:
        - Chunk A: "Sarah Chen leads Acme Corp"
        - Chunk B: "Acme Corp manufactures widgets"
        - Query about Sarah's company's products should find both chunks
        """
        collection = f"test_multihop_{uuid.uuid4().hex[:8]}"
        store = EntityGraphStore(collection=collection)
        try:
            store.clear()

            # Simulate enriched chunks with entities
            store.add_chunk_entities(
                "chunk_sarah",
                [("Sarah Chen", "person"), ("Acme Corp", "company")],
            )
            store.add_chunk_entities(
                "chunk_acme",
                [("Acme Corp", "company"), ("widgets", "product")],
            )
            store.add_chunk_entities(
                "chunk_unrelated",
                [("Bob Smith", "person"), ("Other Inc", "company")],
            )

            # Simulating: vector search found chunk_sarah
            # Entity expansion should also find chunk_acme via "Acme Corp"
            related = store.get_related_chunks(["chunk_sarah"])

            assert "chunk_acme" in related
            assert "chunk_unrelated" not in related
        finally:
            store.clear()
            store.close()

    def test_person_connections(self):
        """Test finding all mentions of a person across documents."""
        collection = f"test_person_{uuid.uuid4().hex[:8]}"
        store = EntityGraphStore(collection=collection)
        try:
            store.clear()

            store.add_chunk_entities("bio", [("John Doe", "person")])
            store.add_chunk_entities(
                "meeting_notes", [("John Doe", "person"), ("Q4 Review", "event")]
            )
            store.add_chunk_entities(
                "email", [("John Doe", "person"), ("Project Alpha", "project")]
            )
            store.add_chunk_entities("other_doc", [("Jane Smith", "person")])

            # Get all chunks mentioning John Doe
            john_chunks = store.get_chunks_for_entity("John Doe")
            assert set(john_chunks) == {"bio", "meeting_notes", "email"}

            # Entity expansion from bio finds related docs
            related = store.get_related_chunks(["bio"])
            assert "meeting_notes" in related
            assert "email" in related
            assert "other_doc" not in related
        finally:
            store.clear()
            store.close()

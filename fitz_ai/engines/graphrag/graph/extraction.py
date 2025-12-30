# fitz_ai/engines/graphrag/graph/extraction.py
"""
Entity and Relationship Extraction for GraphRAG.

Uses LLM to extract entities and relationships from text chunks.
"""

import json
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from fitz_ai.engines.graphrag.config.schema import GraphExtractionConfig
from fitz_ai.engines.graphrag.graph.storage import Entity, KnowledgeGraph, Relationship
from fitz_ai.logging.logger import get_logger

logger = get_logger(__name__)


EXTRACTION_PROMPT = """Extract entities and relationships from the following documents.

## Entity Types to Extract
{entity_types}

## Instructions
1. Identify all important entities (people, organizations, locations, concepts, etc.)
2. For each entity, provide: name, type, description, and which document(s) it appears in
3. Identify relationships between entities
4. For each relationship, provide: source entity, target entity, relationship type, description, and which document(s) it appears in

## Output Format
Return a JSON object with this exact structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "entity_type", "description": "Brief description", "docs": ["doc_id1", "doc_id2"]}}
  ],
  "relationships": [
    {{"source": "Source Entity Name", "target": "Target Entity Name", "type": "relationship_type", "description": "Brief description", "docs": ["doc_id1"]}}
  ]
}}

## Documents to Analyze
{documents}

## Output (JSON only, no other text):"""


class EntityRelationshipExtractor:
    """
    Extracts entities and relationships from text using LLM.

    This is the core extraction component of GraphRAG that builds
    the knowledge graph from document chunks.

    Uses tier="fast" for extraction since this is a background task
    processing many chunks.
    """

    def __init__(
        self,
        config: GraphExtractionConfig,
        llm_provider: Optional[str] = None,
    ):
        """
        Initialize extractor.

        Args:
            config: Extraction configuration
            llm_provider: LLM provider name (uses default from config if None)
        """
        self.config = config
        self._llm_provider = llm_provider or "cohere"
        self._chat_client = None

    def _get_chat_client(self):
        """Lazy load chat client with 'fast' tier for background processing."""
        if self._chat_client is None:
            from fitz_ai.llm.registry import get_llm_plugin

            self._chat_client = get_llm_plugin(
                plugin_type="chat",
                plugin_name=self._llm_provider,
                tier="fast",  # Use fast model for background extraction
            )
        return self._chat_client

    def extract_from_chunks(
        self,
        chunks: List[Dict[str, Any]],
        graph: Optional[KnowledgeGraph] = None,
        batch_size: int = 5,
    ) -> KnowledgeGraph:
        """
        Extract entities and relationships from multiple chunks in batched LLM calls.

        Args:
            chunks: List of chunk dicts with 'text' and optionally 'id' keys
            graph: Optional existing graph to add to
            batch_size: Number of chunks per LLM call

        Returns:
            KnowledgeGraph with extracted entities and relationships
        """
        if graph is None:
            graph = KnowledgeGraph()

        # Filter empty chunks and assign IDs
        valid_chunks = []
        for chunk in chunks:
            text = chunk.get("text", "")
            if text.strip():
                chunk_id = chunk.get("id", str(uuid.uuid4())[:8])
                valid_chunks.append({"id": chunk_id, "text": text})

        if not valid_chunks:
            return graph

        # Process in batches
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i : i + batch_size]
            try:
                entities, relationships = self._extract_batch(batch)
                for entity in entities:
                    graph.add_entity(entity)
                for rel in relationships:
                    graph.add_relationship(rel)
            except Exception as e:
                logger.warning(f"Batch extraction failed: {e}")
                continue

        logger.info(
            f"Extracted {graph.num_entities} entities and "
            f"{graph.num_relationships} relationships from {len(valid_chunks)} chunks"
        )

        return graph

    def _extract_batch(
        self, chunks: List[Dict[str, str]]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Extract from multiple chunks in a single LLM call.

        Args:
            chunks: List of {"id": str, "text": str} dicts

        Returns:
            Tuple of (entities, relationships)
        """
        # Build documents section using basenames for simpler LLM output
        # Map basenames back to full IDs
        basename_to_full_id = {}
        docs_parts = []
        for chunk in chunks:
            full_id = chunk["id"]
            basename = os.path.basename(full_id)
            basename_to_full_id[basename] = full_id
            # Truncate long texts (2000 chars per doc to keep total reasonable)
            text = chunk["text"][:2000]
            docs_parts.append(f"### Document: {basename}\n{text}")
        documents = "\n\n".join(docs_parts)

        # Build prompt
        entity_types_str = (
            ", ".join(self.config.entity_types) if self.config.entity_types else "any"
        )
        prompt = EXTRACTION_PROMPT.format(
            entity_types=entity_types_str,
            documents=documents,
        )

        # Call LLM
        chat_client = self._get_chat_client()
        response = chat_client.chat([{"role": "user", "content": prompt}])

        # Parse response and map basenames back to full IDs
        return self._parse_batch_response(response, basename_to_full_id)

    def _parse_batch_response(
        self, response: str, basename_to_full_id: Dict[str, str]
    ) -> Tuple[List[Entity], List[Relationship]]:
        """Parse LLM response from batched extraction."""
        entities = []
        relationships = []

        # Try to extract JSON from response
        json_match = re.search(r"\{[\s\S]*\}", response)
        if not json_match:
            logger.warning("No JSON found in extraction response")
            return entities, relationships

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse extraction JSON: {e}")
            return entities, relationships

        # Parse entities
        entity_name_to_id: Dict[str, str] = {}
        for e_data in data.get("entities", []):
            if not isinstance(e_data, dict):
                continue

            name = e_data.get("name", "").strip()
            if not name:
                continue

            entity_id = str(uuid.uuid4())[:8]
            entity_name_to_id[name.lower()] = entity_id

            # Get source docs and map basenames to full IDs
            source_docs = e_data.get("docs", [])
            source_chunks = [
                basename_to_full_id[d] for d in source_docs if d in basename_to_full_id
            ]

            entity = Entity(
                id=entity_id,
                name=name,
                type=e_data.get("type", "unknown").lower(),
                description=e_data.get("description", ""),
                source_chunks=source_chunks,
            )
            entities.append(entity)

        # Parse relationships
        for r_data in data.get("relationships", []):
            if not isinstance(r_data, dict):
                continue

            source_name = r_data.get("source", "").strip().lower()
            target_name = r_data.get("target", "").strip().lower()

            source_id = entity_name_to_id.get(source_name)
            target_id = entity_name_to_id.get(target_name)

            if not source_id or not target_id:
                continue

            # Get source docs and map basenames to full IDs
            source_docs = r_data.get("docs", [])
            source_chunks = [
                basename_to_full_id[d] for d in source_docs if d in basename_to_full_id
            ]

            rel = Relationship(
                source_id=source_id,
                target_id=target_id,
                type=r_data.get("type", "related_to").lower().replace(" ", "_"),
                description=r_data.get("description", ""),
                source_chunks=source_chunks,
            )
            relationships.append(rel)

        return entities, relationships

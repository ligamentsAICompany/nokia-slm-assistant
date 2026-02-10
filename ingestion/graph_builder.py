"""
Knowledge Graph Builder (Neo4j)
===============================
Extracts entities and relationships from text and persists them into Neo4j.
Fully replaces the previous NetworkX-based graph builder.
"""

import re
import logging
import sys
from typing import List, Set, Tuple, Dict
from pathlib import Path

from .chunker import ChunkData

# Import Neo4j helpers from the top-level module
sys.path.insert(0, str(Path(__file__).parent.parent))

from neo4j_graph import (
    get_connection, init_graph, ensure_schema,
    create_concept, create_alarm, create_entity, create_document,
    relate_concepts, link_concept_to_document, link_alarm_to_entity,
    link_alarm_to_concept, link_entity_part_of,
    get_graph_stats as neo4j_get_graph_stats,
    seed_gpon_knowledge,
)

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a Neo4j knowledge graph from document chunks.

    Extracts Nokia-specific entities and relationships using regex patterns.
    All data is persisted directly into Neo4j via Bolt.

    Usage:
        builder = GraphBuilder()
        builder.process_chunks(chunks, doc_id="my_doc")
        builder.save()  # no-op – Neo4j is already persistent
    """

    # === Entity Patterns ===
    ENTITY_PATTERNS = {
        "COMPONENT": [
            (r"\bONT[-\s]?\d*\b", "ONT"),
            (r"\bONU[-\s]?\d*\b", "ONU"),
            (r"\bOLT[-\s]?\d*\b", "OLT"),
            (r"\bISAM\b", "ISAM"),
            (r"\bPON\s*(?:\[\d+/\d+/\d+\])?\b", "PON"),
            (r"\bGPON\b", "GPON"),
            (r"\bXGS-PON\b", "XGS-PON"),
            (r"\bSplitter\b", "Splitter"),
            (r"\bSFP\b", "SFP"),
        ],
        "PROTOCOL": [
            (r"\bT-CONT\b", "T-CONT"),
            (r"\bGEM\s?(?:Port)?\b", "GEM"),
            (r"\bDBA\b", "DBA"),
            (r"\bPLOAM\b", "PLOAM"),
            (r"\bOMCI\b", "OMCI"),
            (r"\bTR-069\b", "TR-069"),
        ],
        "FEATURE": [
            (r"\bVLAN\b", "VLAN"),
            (r"\bQoS\b", "QoS"),
            (r"\bMGMT\b", "MGMT"),
            (r"\bAES\b", "AES"),
            (r"\bFEC\b", "FEC"),
            (r"\bBandwidth\s+Profile\b", "Bandwidth Profile"),
            (r"\bAlloc-ID\b", "Alloc-ID"),
            (r"\bService\s+Port\b", "Service Port"),
        ],
        "ALARM": [
            (r"\bLOS\b", "LOS"),
            (r"\bLOF\b", "LOF"),
            (r"\bSF\b", "SF"),
            (r"\bSD\b", "SD"),
            (r"\bSUF\b", "SUF"),
            (r"\bDying[\s_]?Gasp\b", "Dying Gasp"),
            (r"\bLOM\b", "LOM"),
        ],
        "COMMAND": [
            (r"\bconfigure\s+\w+", "Configure Command"),
            (r"\bshow\s+\w+", "Show Command"),
            (r"\bno\s+\w+", "No Command"),
        ],
    }

    # === Relationship Patterns ===
    RELATIONSHIP_PATTERNS = [
        (r"(\w+)\s+(?:is\s+)?connect(?:ed|s)?\s+(?:to|with)\s+(\w+)", "CONNECTS_TO"),
        (r"(\w+)\s+(?:is\s+)?part\s+of\s+(\w+)", "PART_OF"),
        (r"(\w+)\s+(?:is\s+)?used\s+(?:by|for)\s+(\w+)", "USED_BY"),
        (r"(\w+)\s+(?:is\s+)?related\s+to\s+(\w+)", "RELATED_TO"),
        (r"(\w+)\s+(?:causes?|triggers?)\s+(\w+)", "CAUSES"),
        (r"(\w+)\s+(?:supports?|enables?)\s+(\w+)", "ENABLES"),
    ]

    # Canonical name sets for routing to the correct Neo4j node label
    _ALARM_NAMES = {"LOS", "LOF", "SF", "SD", "SUF", "Dying Gasp", "LOM"}
    _ENTITY_NAMES = {"ONT", "ONU", "OLT", "ISAM", "PON", "GPON",
                     "XGS-PON", "Splitter", "SFP"}

    def __init__(self):
        """Initialize graph builder – connects to Neo4j."""
        self._conn = get_connection()
        if not self._conn.connected:
            self._conn.connect()

        ensure_schema(self._conn)

        self._entity_cache: Set[str] = set()
        self._counts = {
            "concepts": 0, "alarms": 0, "entities": 0,
            "documents": 0, "relationships": 0,
        }

        # Compile patterns
        self._entity_compiled = {}
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            self._entity_compiled[entity_type] = [
                (re.compile(p, re.IGNORECASE), name) for p, name in patterns
            ]

        self._rel_compiled = [
            (re.compile(p, re.IGNORECASE), rel_type)
            for p, rel_type in self.RELATIONSHIP_PATTERNS
        ]

    # -----------------------------------------------------------------
    # Extraction
    # -----------------------------------------------------------------

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """Extract entities from text. Returns [(name, category_type)]."""
        entities = []
        for entity_type, patterns in self._entity_compiled.items():
            for pattern, name in patterns:
                if pattern.search(text):
                    entities.append((name, entity_type))
        return list(set(entities))

    def extract_relationships(
        self, text: str, known_entities: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """Extract relationships from text."""
        relationships = []
        for pattern, rel_type in self._rel_compiled:
            for match in pattern.finditer(text):
                source = match.group(1).strip()
                target = match.group(2).strip()
                if source in known_entities or target in known_entities:
                    relationships.append((source, target, rel_type))
        return relationships

    # -----------------------------------------------------------------
    # Persistence helpers
    # -----------------------------------------------------------------

    def _persist_entity(self, name: str, entity_type: str, source_page: int = 0):
        """Write a single entity to Neo4j based on its type."""
        if name in self._ALARM_NAMES:
            if create_alarm(self._conn, name, "unknown"):
                self._counts["alarms"] += 1
        elif name in self._ENTITY_NAMES:
            if create_entity(self._conn, name, entity_type.lower()):
                self._counts["entities"] += 1
        else:
            if create_concept(self._conn, name, entity_type.lower()):
                self._counts["concepts"] += 1
        self._entity_cache.add(name)

    def _persist_relationship(self, source: str, target: str, rel_type: str):
        """Write a relationship to Neo4j."""
        if source in self._ALARM_NAMES and target in self._ENTITY_NAMES:
            link_alarm_to_entity(self._conn, source, target)
        elif source in self._ALARM_NAMES:
            link_alarm_to_concept(self._conn, source, target)
        elif rel_type == "PART_OF" and source in self._ENTITY_NAMES and target in self._ENTITY_NAMES:
            link_entity_part_of(self._conn, source, target)
        else:
            relate_concepts(self._conn, source, target)
        self._counts["relationships"] += 1

    # -----------------------------------------------------------------
    # Public add helpers (kept for API compatibility)
    # -----------------------------------------------------------------

    def add_entity(self, name: str, entity_type: str, **attributes):
        """Add an entity node to Neo4j."""
        self._persist_entity(name, entity_type, source_page=attributes.get("source_page", 0))
        logger.debug(f"Added entity: {name} ({entity_type})")

    def add_relationship(self, source: str, target: str, rel_type: str, **attributes):
        """Add a relationship to Neo4j."""
        if source not in self._entity_cache:
            self.add_entity(source, "UNKNOWN")
        if target not in self._entity_cache:
            self.add_entity(target, "UNKNOWN")
        self._persist_relationship(source, target, rel_type)
        logger.debug(f"Added relationship: {source} -[{rel_type}]-> {target}")

    # -----------------------------------------------------------------
    # Chunk processing
    # -----------------------------------------------------------------

    def process_chunk(self, chunk: ChunkData, doc_id: str = ""):
        """Process a single chunk – extract, persist, link to document."""
        entities = self.extract_entities(chunk.text)
        for name, entity_type in entities:
            self._persist_entity(name, entity_type, source_page=chunk.page_number)

        # Link entities/concepts to their source document page
        if doc_id:
            for name, entity_type in entities:
                if name not in self._ALARM_NAMES:
                    link_concept_to_document(self._conn, name, doc_id, chunk.page_number)
                    self._counts["documents"] += 1

        # Extract & persist relationships
        entity_names = {e[0] for e in entities}
        relationships = self.extract_relationships(chunk.text, entity_names)
        for source, target, rel_type in relationships:
            if source not in self._entity_cache:
                self._persist_entity(source, "UNKNOWN")
            if target not in self._entity_cache:
                self._persist_entity(target, "UNKNOWN")
            self._persist_relationship(source, target, rel_type)

    def process_chunks(
        self,
        chunks: List[ChunkData],
        doc_id: str = "unknown_doc",
        show_progress: bool = True,
    ):
        """Process multiple chunks."""
        if show_progress:
            logger.info(f"Processing {len(chunks)} chunks for Neo4j graph extraction...")

        for i, chunk in enumerate(chunks):
            self.process_chunk(chunk, doc_id=doc_id)

            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"  Processed {i + 1}/{len(chunks)} chunks")

        if show_progress:
            stats = self.get_stats()
            logger.info(
                f"Graph extraction complete. "
                f"Nodes: {stats['total_nodes']}, Rels: {stats['total_relationships']}"
            )

    def add_predefined_relationships(self):
        """Seed foundational GPON domain knowledge into Neo4j."""
        seed_gpon_knowledge(self._conn)

    # -----------------------------------------------------------------
    # Stats / save  (save is a no-op; Neo4j is already persistent)
    # -----------------------------------------------------------------

    def save(self, path: str = "", backup: bool = True):
        """No-op.  Neo4j persists automatically.  Kept for API compatibility."""
        logger.info("[GraphBuilder] Data already persisted in Neo4j (no file save needed)")

    @classmethod
    def load(cls, path: str = "") -> "GraphBuilder":
        """Return a new builder connected to the existing Neo4j graph."""
        return cls()

    def get_stats(self) -> Dict:
        """Get graph statistics from Neo4j."""
        neo_stats = neo4j_get_graph_stats(self._conn)
        return {
            "nodes": neo_stats.get("total_nodes", 0),
            "edges": neo_stats.get("total_relationships", 0),
            "total_nodes": neo_stats.get("total_nodes", 0),
            "total_relationships": neo_stats.get("total_relationships", 0),
            "node_types": neo_stats.get("nodes_by_label", {}),
            "edge_types": neo_stats.get("relationships_by_type", {}),
            "ingestion_counts": self._counts,
        }

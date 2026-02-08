"""
Knowledge Graph Builder
=======================
Extracts entities and relationships from text to build a knowledge graph.
"""

import re
import logging
from typing import List, Set, Tuple, Dict, Optional
from pathlib import Path

from .chunker import ChunkData

logger = logging.getLogger(__name__)


class GraphBuilder:
    """
    Builds a NetworkX knowledge graph from document chunks.
    
    Extracts Nokia-specific entities and relationships using regex patterns.
    No LLM usage during extraction.
    
    Usage:
        builder = GraphBuilder()
        builder.process_chunks(chunks)
        builder.save("path/to/graph.gml")
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
        ],
        "ALARM": [
            (r"\bLOS\b", "LOS"),
            (r"\bLOF\b", "LOF"),
            (r"\bSF\b", "SF"),
            (r"\bSD\b", "SD"),
            (r"\bDying[\s_]?Gasp\b", "Dying Gasp"),
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
    
    def __init__(self):
        """Initialize graph builder."""
        import networkx as nx
        self.graph = nx.DiGraph()
        self._entity_cache: Set[str] = set()
        
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
    
    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """
        Extract entities from text.
        
        Args:
            text: Text to extract from.
            
        Returns:
            List of (entity_name, entity_type) tuples.
        """
        entities = []
        
        for entity_type, patterns in self._entity_compiled.items():
            for pattern, name in patterns:
                if pattern.search(text):
                    entities.append((name, entity_type))
        
        return list(set(entities))
    
    def extract_relationships(
        self, 
        text: str, 
        known_entities: Set[str]
    ) -> List[Tuple[str, str, str]]:
        """
        Extract relationships from text.
        
        Args:
            text: Text to extract from.
            known_entities: Set of known entity names for filtering.
            
        Returns:
            List of (source, target, relationship_type) tuples.
        """
        relationships = []
        
        for pattern, rel_type in self._rel_compiled:
            for match in pattern.finditer(text):
                source = match.group(1).strip()
                target = match.group(2).strip()
                
                # Only include if at least one is a known entity
                if source in known_entities or target in known_entities:
                    relationships.append((source, target, rel_type))
        
        return relationships
    
    def add_entity(self, name: str, entity_type: str, **attributes):
        """
        Add an entity node to the graph.
        
        Args:
            name: Entity name.
            entity_type: Entity type.
            **attributes: Additional node attributes.
        """
        if name not in self.graph:
            self.graph.add_node(name, type=entity_type, **attributes)
            self._entity_cache.add(name)
            logger.debug(f"Added entity: {name} ({entity_type})")
    
    def add_relationship(
        self, 
        source: str, 
        target: str, 
        rel_type: str,
        **attributes
    ):
        """
        Add a relationship edge to the graph.
        
        Args:
            source: Source entity name.
            target: Target entity name.
            rel_type: Relationship type.
            **attributes: Additional edge attributes.
        """
        # Ensure both nodes exist
        if source not in self.graph:
            self.add_entity(source, "UNKNOWN")
        if target not in self.graph:
            self.add_entity(target, "UNKNOWN")
        
        self.graph.add_edge(source, target, type=rel_type, **attributes)
        logger.debug(f"Added relationship: {source} -[{rel_type}]-> {target}")
    
    def process_chunk(self, chunk: ChunkData):
        """
        Process a single chunk and extract entities/relationships.
        
        Args:
            chunk: ChunkData to process.
        """
        # Extract entities
        entities = self.extract_entities(chunk.text)
        for name, entity_type in entities:
            self.add_entity(name, entity_type, source_page=chunk.page_number)
        
        # Extract relationships
        entity_names = {e[0] for e in entities}
        relationships = self.extract_relationships(chunk.text, entity_names)
        for source, target, rel_type in relationships:
            self.add_relationship(source, target, rel_type)
    
    def process_chunks(self, chunks: List[ChunkData], show_progress: bool = True):
        """
        Process multiple chunks.
        
        Args:
            chunks: List of ChunkData to process.
            show_progress: Whether to log progress.
        """
        if show_progress:
            logger.info(f"Processing {len(chunks)} chunks for graph extraction...")
        
        for i, chunk in enumerate(chunks):
            self.process_chunk(chunk)
            
            if show_progress and (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(chunks)} chunks")
        
        if show_progress:
            logger.info(
                f"Graph extraction complete. "
                f"Nodes: {self.graph.number_of_nodes()}, "
                f"Edges: {self.graph.number_of_edges()}"
            )
    
    def add_predefined_relationships(self):
        """Add known Nokia-specific relationships."""
        # Core GPON hierarchy
        hierarchy = [
            ("OLT", "PON", "CONTAINS"),
            ("PON", "ONT", "CONNECTS_TO"),
            ("ONT", "GEM", "USES"),
            ("ONT", "T-CONT", "USES"),
            ("T-CONT", "DBA", "MANAGED_BY"),
            ("GEM", "QoS", "APPLIES"),
        ]
        
        for source, target, rel_type in hierarchy:
            if source in self._entity_cache or target in self._entity_cache:
                self.add_relationship(source, target, rel_type)
        
        # Alarm relationships
        alarm_rels = [
            ("LOS", "ONT", "AFFECTS"),
            ("LOF", "PON", "AFFECTS"),
            ("Dying Gasp", "ONT", "INDICATES"),
        ]
        
        for alarm, target, rel_type in alarm_rels:
            if alarm in self._entity_cache:
                self.add_relationship(alarm, target, rel_type)
    
    def save(self, path: str, backup: bool = True):
        """
        Save graph to GML file.
        
        Args:
            path: Output path.
            backup: Whether to backup existing file.
        """
        import networkx as nx
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        if backup and path.exists():
            backup_path = path.with_suffix('.gml.backup')
            import shutil
            shutil.copy2(path, backup_path)
            logger.info(f"Backed up existing graph to {backup_path}")
        
        nx.write_gml(self.graph, str(path))
        logger.info(
            f"Saved graph ({self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges) to {path}"
        )
    
    @classmethod
    def load(cls, path: str) -> 'GraphBuilder':
        """
        Load graph from GML file.
        
        Args:
            path: Path to GML file.
            
        Returns:
            GraphBuilder instance.
        """
        import networkx as nx
        
        builder = cls()
        builder.graph = nx.read_gml(str(path))
        builder._entity_cache = set(builder.graph.nodes())
        
        logger.info(f"Loaded graph with {builder.graph.number_of_nodes()} nodes")
        
        return builder
    
    def get_stats(self) -> Dict:
        """Get graph statistics."""
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": self._count_by_attribute("type", nodes=True),
            "edge_types": self._count_by_attribute("type", nodes=False),
        }
    
    def _count_by_attribute(self, attr: str, nodes: bool = True) -> Dict[str, int]:
        """Count elements by attribute value."""
        from collections import defaultdict
        counts = defaultdict(int)
        
        if nodes:
            for _, data in self.graph.nodes(data=True):
                counts[data.get(attr, "UNKNOWN")] += 1
        else:
            for _, _, data in self.graph.edges(data=True):
                counts[data.get(attr, "UNKNOWN")] += 1
        
        return dict(counts)

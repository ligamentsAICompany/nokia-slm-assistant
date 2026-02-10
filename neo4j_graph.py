"""
Neo4j Knowledge Graph Module
=============================
Singleton Neo4j driver and graph operations for the Nokia GPON SLM system.
Replaces all NetworkX graph functionality with persistent Neo4j storage.

Schema:
  (:Concept {name, type})
  (:Alarm {code, severity, description})
  (:Entity {name, category})
  (:Document {doc_id, page, source})

  (Concept)-[:RELATED_TO]->(Concept)
  (Concept)-[:MENTIONED_IN]->(Document)
  (Alarm)-[:AFFECTS]->(Entity)
  (Alarm)-[:RELATED_TO]->(Concept)
  (Entity)-[:PART_OF]->(Entity)
"""

import os
import time
import logging
import threading
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-load Neo4j driver at module level
# ---------------------------------------------------------------------------
try:
    from neo4j import GraphDatabase
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    logger.warning("[neo4j_graph] neo4j driver not installed – graph features disabled")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Credentials from environment
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "nokia-slm-2026")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")


# =============================================================================
# SINGLETON DRIVER
# =============================================================================

class Neo4jConnection:
    """Thread-safe singleton Neo4j driver wrapper."""

    _instance: Optional["Neo4jConnection"] = None
    _lock = threading.Lock()

    def __init__(self):
        self._driver = None

    # -- Singleton accessor --------------------------------------------------
    @classmethod
    def get_instance(cls) -> "Neo4jConnection":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    # -- Connection management -----------------------------------------------
    def connect(self) -> bool:
        """Open the driver connection.  Returns True on success."""
        if not NEO4J_AVAILABLE:
            logger.error("[neo4j_graph] neo4j package not installed")
            return False
        try:
            self._driver = GraphDatabase.driver(
                NEO4J_URI,
                auth=(NEO4J_USER, NEO4J_PASSWORD),
            )
            self._driver.verify_connectivity()
            logger.info(f"[neo4j_graph] Connected to {NEO4J_URI}")
            return True
        except Exception as exc:
            logger.error(f"[neo4j_graph] Connection failed: {exc}")
            self._driver = None
            return False

    def close(self):
        """Close the driver connection."""
        if self._driver:
            self._driver.close()
            self._driver = None
            logger.info("[neo4j_graph] Connection closed")

    @property
    def driver(self):
        return self._driver

    @property
    def connected(self) -> bool:
        return self._driver is not None


# =============================================================================
# SCHEMA INITIALISATION  (idempotent)
# =============================================================================

SCHEMA_CYPHER = [
    # Uniqueness constraints (also create implicit indexes)
    "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Alarm)   REQUIRE a.code IS UNIQUE",
    "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity)  REQUIRE e.name IS UNIQUE",
    # Index for Document lookup
    "CREATE INDEX IF NOT EXISTS FOR (d:Document) ON (d.doc_id, d.page)",
]


def ensure_schema(conn: Neo4jConnection) -> bool:
    """Create constraints/indexes if they do not exist."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as session:
            for stmt in SCHEMA_CYPHER:
                session.run(stmt)
        logger.info("[neo4j_graph] Schema constraints ensured")
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] Schema creation error: {exc}")
        return False


# =============================================================================
# WRITE HELPERS  (all idempotent via MERGE)
# =============================================================================

def create_concept(conn: Neo4jConnection, name: str, concept_type: str = "general") -> bool:
    """MERGE a Concept node."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (c:Concept {name: $name}) "
                "ON CREATE SET c.type = $type, c.created_at = datetime() "
                "ON MATCH  SET c.type = $type",
                name=name, type=concept_type,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] create_concept error: {exc}")
        return False


def create_alarm(conn: Neo4jConnection, code: str, severity: str = "unknown",
                 description: str = "") -> bool:
    """MERGE an Alarm node."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (a:Alarm {code: $code}) "
                "ON CREATE SET a.severity = $sev, a.description = $desc, a.created_at = datetime() "
                "ON MATCH  SET a.severity = $sev, a.description = $desc",
                code=code, sev=severity, desc=description,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] create_alarm error: {exc}")
        return False


def create_entity(conn: Neo4jConnection, name: str, category: str = "component") -> bool:
    """MERGE an Entity node."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (e:Entity {name: $name}) "
                "ON CREATE SET e.category = $cat, e.created_at = datetime() "
                "ON MATCH  SET e.category = $cat",
                name=name, cat=category,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] create_entity error: {exc}")
        return False


def create_document(conn: Neo4jConnection, doc_id: str, page: int,
                    source: str = "") -> bool:
    """MERGE a Document node."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (d:Document {doc_id: $doc, page: $page}) "
                "ON CREATE SET d.source = $src, d.created_at = datetime()",
                doc=doc_id, page=page, src=source,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] create_document error: {exc}")
        return False


# ---------------------------------------------------------------------------
# Relationship helpers
# ---------------------------------------------------------------------------

def relate_concepts(conn: Neo4jConnection, concept_a: str, concept_b: str,
                    weight: float = 1.0) -> bool:
    """(Concept)-[:RELATED_TO]->(Concept)  bidirectional via two MERGEs."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (a:Concept {name: $a}) "
                "MERGE (b:Concept {name: $b}) "
                "MERGE (a)-[r:RELATED_TO]->(b) "
                "ON CREATE SET r.weight = $w, r.created_at = datetime() "
                "ON MATCH  SET r.weight = $w",
                a=concept_a, b=concept_b, w=weight,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] relate_concepts error: {exc}")
        return False


def link_concept_to_document(conn: Neo4jConnection, concept: str,
                             doc_id: str, page: int) -> bool:
    """(Concept)-[:MENTIONED_IN]->(Document)."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (c:Concept {name: $c}) "
                "MERGE (d:Document {doc_id: $doc, page: $page}) "
                "MERGE (c)-[r:MENTIONED_IN]->(d) "
                "ON CREATE SET r.created_at = datetime()",
                c=concept, doc=doc_id, page=page,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] link_concept_to_document error: {exc}")
        return False


def link_alarm_to_entity(conn: Neo4jConnection, alarm_code: str,
                         entity_name: str) -> bool:
    """(Alarm)-[:AFFECTS]->(Entity)."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (a:Alarm {code: $alarm}) "
                "MERGE (e:Entity {name: $entity}) "
                "MERGE (a)-[r:AFFECTS]->(e) "
                "ON CREATE SET r.created_at = datetime()",
                alarm=alarm_code, entity=entity_name,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] link_alarm_to_entity error: {exc}")
        return False


def link_alarm_to_concept(conn: Neo4jConnection, alarm_code: str,
                          concept_name: str) -> bool:
    """(Alarm)-[:RELATED_TO]->(Concept)."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (a:Alarm {code: $alarm}) "
                "MERGE (c:Concept {name: $concept}) "
                "MERGE (a)-[r:RELATED_TO]->(c) "
                "ON CREATE SET r.created_at = datetime()",
                alarm=alarm_code, concept=concept_name,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] link_alarm_to_concept error: {exc}")
        return False


def link_entity_part_of(conn: Neo4jConnection, child: str, parent: str) -> bool:
    """(Entity)-[:PART_OF]->(Entity)."""
    if not conn.connected:
        return False
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            s.run(
                "MERGE (a:Entity {name: $child}) "
                "MERGE (b:Entity {name: $parent}) "
                "MERGE (a)-[r:PART_OF]->(b) "
                "ON CREATE SET r.created_at = datetime()",
                child=child, parent=parent,
            )
        return True
    except Exception as exc:
        logger.error(f"[neo4j_graph] link_entity_part_of error: {exc}")
        return False


# =============================================================================
# READ / QUERY HELPERS  (parameterised Cypher)
# =============================================================================

def query_related_concepts(conn: Neo4jConnection, term: str,
                           max_hops: int = 2, limit: int = 15) -> List[Dict]:
    """
    Find concepts related to *term* within *max_hops* traversal.
    Returns list of {name, type, relationship, distance}.
    """
    if not conn.connected:
        return []
    try:
        start = time.time()
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            result = s.run(
                """
                MATCH (start:Concept)
                WHERE toLower(start.name) CONTAINS toLower($term)
                CALL {
                    WITH start
                    MATCH path = (start)-[:RELATED_TO*1..2]-(related:Concept)
                    WHERE related <> start
                    RETURN related, length(path) AS dist
                    ORDER BY dist
                    LIMIT $lim
                }
                RETURN DISTINCT related.name AS name,
                       related.type  AS type,
                       dist          AS distance
                ORDER BY distance
                """,
                term=term, lim=limit,
            )
            records = [dict(r) for r in result]
        elapsed = (time.time() - start) * 1000
        logger.debug(f"[neo4j_graph] query_related_concepts({term}): {len(records)} results in {elapsed:.1f}ms")
        return records
    except Exception as exc:
        logger.error(f"[neo4j_graph] query_related_concepts error: {exc}")
        return []


def query_alarm_context(conn: Neo4jConnection, alarm_code: str) -> List[Dict]:
    """
    Return alarm context: affected entities and related concepts.
    """
    if not conn.connected:
        return []
    try:
        start = time.time()
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            result = s.run(
                """
                MATCH (a:Alarm)
                WHERE toLower(a.code) = toLower($code)
                OPTIONAL MATCH (a)-[:AFFECTS]->(e:Entity)
                OPTIONAL MATCH (a)-[:RELATED_TO]->(c:Concept)
                RETURN a.code       AS alarm,
                       a.severity   AS severity,
                       a.description AS description,
                       collect(DISTINCT e.name) AS affected_entities,
                       collect(DISTINCT c.name) AS related_concepts
                """,
                code=alarm_code,
            )
            records = [dict(r) for r in result]
        elapsed = (time.time() - start) * 1000
        logger.debug(f"[neo4j_graph] query_alarm_context({alarm_code}): {len(records)} in {elapsed:.1f}ms")
        return records
    except Exception as exc:
        logger.error(f"[neo4j_graph] query_alarm_context error: {exc}")
        return []


def query_entity_context(conn: Neo4jConnection, entity_name: str) -> List[Dict]:
    """
    Return entity hierarchy and related alarms.
    """
    if not conn.connected:
        return []
    try:
        start = time.time()
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            result = s.run(
                """
                MATCH (e:Entity)
                WHERE toLower(e.name) CONTAINS toLower($name)
                OPTIONAL MATCH (e)-[:PART_OF]->(parent:Entity)
                OPTIONAL MATCH (child:Entity)-[:PART_OF]->(e)
                OPTIONAL MATCH (alarm:Alarm)-[:AFFECTS]->(e)
                RETURN e.name        AS entity,
                       e.category    AS category,
                       collect(DISTINCT parent.name) AS parents,
                       collect(DISTINCT child.name)  AS children,
                       collect(DISTINCT alarm.code)  AS alarms
                """,
                name=entity_name,
            )
            records = [dict(r) for r in result]
        elapsed = (time.time() - start) * 1000
        logger.debug(f"[neo4j_graph] query_entity_context({entity_name}): {len(records)} in {elapsed:.1f}ms")
        return records
    except Exception as exc:
        logger.error(f"[neo4j_graph] query_entity_context error: {exc}")
        return []


def query_concept_documents(conn: Neo4jConnection, concept: str,
                            limit: int = 10) -> List[Dict]:
    """
    Return documents where a concept is mentioned.
    """
    if not conn.connected:
        return []
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            result = s.run(
                """
                MATCH (c:Concept)-[:MENTIONED_IN]->(d:Document)
                WHERE toLower(c.name) CONTAINS toLower($concept)
                RETURN d.doc_id AS doc_id, d.page AS page, d.source AS source
                ORDER BY d.page
                LIMIT $lim
                """,
                concept=concept, lim=limit,
            )
            return [dict(r) for r in result]
    except Exception as exc:
        logger.error(f"[neo4j_graph] query_concept_documents error: {exc}")
        return []


# =============================================================================
# GRAPH SEARCH FOR RAG PIPELINE
# =============================================================================

def graph_search_for_query(conn: Neo4jConnection, query: str) -> str:
    """
    High-level graph search used during RAG retrieval.
    Extracts keywords from the query, searches Neo4j for related concepts,
    alarms, and entities, then formats the results as context text.

    Returns a formatted string or empty string if no graph data found.
    """
    if not conn.connected:
        return ""

    start = time.time()
    keywords = _extract_keywords(query)
    if not keywords:
        return ""

    context_lines: List[str] = []
    nodes_matched = 0
    edges_matched = 0

    # 1. Search for related concepts
    for kw in keywords[:5]:  # cap to 5 keywords
        related = query_related_concepts(conn, kw, max_hops=2, limit=8)
        for r in related:
            nodes_matched += 1
            edges_matched += 1
            context_lines.append(
                f"Concept '{kw}' is related to '{r['name']}' (type: {r.get('type','?')}, distance: {r.get('distance',1)})"
            )

    # 2. Search for alarms
    alarm_codes = _extract_alarm_codes(query)
    for code in alarm_codes:
        alarm_data = query_alarm_context(conn, code)
        for ad in alarm_data:
            if ad.get("alarm"):
                nodes_matched += 1
                entities = ", ".join(ad.get("affected_entities", []) or ["unknown"])
                concepts = ", ".join(ad.get("related_concepts", []) or [])
                sev = ad.get("severity", "unknown")
                line = f"Alarm {ad['alarm']} (severity: {sev}) affects: {entities}"
                if concepts:
                    line += f" | related concepts: {concepts}"
                context_lines.append(line)
                edges_matched += len(ad.get("affected_entities", []))

    # 3. Search for entity hierarchy
    entity_names = _extract_entity_names(query)
    for ename in entity_names[:3]:
        ectx = query_entity_context(conn, ename)
        for ec in ectx:
            if ec.get("entity"):
                nodes_matched += 1
                parts = []
                if ec.get("parents"):
                    parts.append(f"part of {', '.join(ec['parents'])}")
                if ec.get("children"):
                    parts.append(f"contains {', '.join(ec['children'])}")
                if ec.get("alarms"):
                    parts.append(f"alarms: {', '.join(ec['alarms'])}")
                if parts:
                    context_lines.append(f"Entity '{ec['entity']}': {'; '.join(parts)}")
                    edges_matched += len(ec.get("children", [])) + len(ec.get("alarms", []))

    elapsed_ms = (time.time() - start) * 1000

    # De-duplicate
    context_lines = list(dict.fromkeys(context_lines))

    if not context_lines:
        logger.debug(f"[neo4j_graph] graph_search: no results for query ({elapsed_ms:.1f}ms)")
        return ""

    header = "--- Knowledge Graph Context (Neo4j) ---"
    body = "\n".join(context_lines)
    footer = f"[Graph: {nodes_matched} nodes, {edges_matched} edges matched in {elapsed_ms:.1f}ms]"

    logger.info(
        f"[neo4j_graph] graph_search: {nodes_matched} nodes, {edges_matched} edges, {elapsed_ms:.1f}ms"
    )
    return f"{header}\n{body}\n{footer}"


# =============================================================================
# GRAPH STATISTICS (for /api/metrics and /api/stats)
# =============================================================================

def get_graph_stats(conn: Neo4jConnection) -> Dict:
    """Return graph-wide statistics for observability."""
    if not conn.connected:
        return {"connected": False}
    try:
        with conn.driver.session(database=NEO4J_DATABASE) as s:
            res = s.run(
                """
                MATCH (n)
                WITH labels(n) AS lbls, count(*) AS cnt
                RETURN lbls, cnt
                """
            )
            label_counts = {}
            total_nodes = 0
            for r in res:
                label = ":".join(r["lbls"]) if r["lbls"] else "unlabeled"
                label_counts[label] = r["cnt"]
                total_nodes += r["cnt"]

            res2 = s.run("MATCH ()-[r]->() RETURN type(r) AS t, count(*) AS cnt")
            rel_counts = {}
            total_rels = 0
            for r in res2:
                rel_counts[r["t"]] = r["cnt"]
                total_rels += r["cnt"]

        return {
            "connected": True,
            "total_nodes": total_nodes,
            "total_relationships": total_rels,
            "nodes_by_label": label_counts,
            "relationships_by_type": rel_counts,
        }
    except Exception as exc:
        logger.error(f"[neo4j_graph] get_graph_stats error: {exc}")
        return {"connected": True, "error": str(exc)}


# =============================================================================
# SEED DATA – GPON domain knowledge
# =============================================================================

def seed_gpon_knowledge(conn: Neo4jConnection) -> Dict:
    """
    Populate foundational GPON knowledge that augments PDF-extracted data.
    All data here is publicly documented Nokia / ITU-T standard information.
    Idempotent (uses MERGE).
    """
    if not conn.connected:
        return {"seeded": False}

    counts = {"concepts": 0, "entities": 0, "alarms": 0, "relationships": 0}

    # -- Concepts --
    concepts = [
        ("T-CONT", "qos"), ("GEM Port", "identifier"),
        ("DBA", "protocol"), ("OMCI", "protocol"), ("PLOAM", "protocol"),
        ("VLAN", "networking"), ("QoS", "qos"), ("Bandwidth Profile", "qos"),
        ("Alloc-ID", "identifier"), ("Service Port", "networking"),
        ("FEC", "feature"), ("AES", "feature"), ("TR-069", "protocol"),
        ("Ranging", "procedure"), ("Activation", "procedure"),
        ("Deactivation", "procedure"), ("GPON", "technology"),
        ("XGS-PON", "technology"), ("Upstream", "direction"),
        ("Downstream", "direction"), ("Optical Power", "measurement"),
        ("SFP", "component"), ("Splitter", "component"),
    ]
    for name, ctype in concepts:
        if create_concept(conn, name, ctype):
            counts["concepts"] += 1

    # -- Entities --
    entities = [
        ("OLT", "network_element"), ("ONT", "network_element"),
        ("ONU", "network_element"), ("PON", "interface"),
        ("ISAM", "platform"), ("7360 ISAM FX", "platform"),
        ("7362 ISAM SX", "platform"), ("G-240W-F", "ont_model"),
    ]
    for name, cat in entities:
        if create_entity(conn, name, cat):
            counts["entities"] += 1

    # -- Alarms --
    alarms = [
        ("LOS", "critical", "Loss of Signal"),
        ("LOF", "critical", "Loss of Frame"),
        ("SF", "major", "Signal Fail"),
        ("SD", "minor", "Signal Degrade"),
        ("SUF", "major", "Startup Failure"),
        ("Dying Gasp", "critical", "ONT power loss notification"),
        ("MEM", "warning", "Memory usage alarm"),
        ("LOM", "major", "Loss of Management"),
    ]
    for code, sev, desc in alarms:
        if create_alarm(conn, code, sev, desc):
            counts["alarms"] += 1

    # -- Concept relationships --
    concept_rels = [
        ("T-CONT", "DBA"), ("T-CONT", "GEM Port"), ("T-CONT", "Alloc-ID"),
        ("GEM Port", "VLAN"), ("GEM Port", "QoS"), ("GEM Port", "Service Port"),
        ("DBA", "Bandwidth Profile"), ("DBA", "Upstream"),
        ("OMCI", "ONT"), ("PLOAM", "Ranging"), ("PLOAM", "Activation"),
        ("GPON", "XGS-PON"), ("GPON", "Downstream"), ("GPON", "Upstream"),
        ("QoS", "Bandwidth Profile"), ("VLAN", "Service Port"),
        ("FEC", "Optical Power"), ("AES", "GPON"),
        ("TR-069", "ONT"), ("Ranging", "Activation"),
        ("Optical Power", "SFP"),
    ]
    for a, b in concept_rels:
        if relate_concepts(conn, a, b):
            counts["relationships"] += 1

    # -- Alarm → Entity --
    alarm_entity = [
        ("LOS", "ONT"), ("LOS", "PON"), ("LOF", "PON"), ("LOF", "OLT"),
        ("SF", "PON"), ("SD", "PON"), ("SUF", "ONT"),
        ("Dying Gasp", "ONT"), ("LOM", "ONT"), ("MEM", "ONT"),
    ]
    for alarm, ent in alarm_entity:
        if link_alarm_to_entity(conn, alarm, ent):
            counts["relationships"] += 1

    # -- Alarm → Concept --
    alarm_concept = [
        ("LOS", "Optical Power"), ("LOS", "SFP"), ("LOS", "Ranging"),
        ("LOF", "PLOAM"), ("SUF", "Activation"), ("SUF", "Ranging"),
        ("SD", "FEC"), ("SF", "FEC"),
    ]
    for alarm, concept in alarm_concept:
        if link_alarm_to_concept(conn, alarm, concept):
            counts["relationships"] += 1

    # -- Entity hierarchy --
    hierarchy = [
        ("PON", "OLT"), ("ONT", "PON"), ("ONU", "PON"),
        ("Splitter", "PON"), ("SFP", "OLT"),
        ("G-240W-F", "ONT"),
    ]
    for child, parent in hierarchy:
        if link_entity_part_of(conn, child, parent):
            counts["relationships"] += 1

    logger.info(f"[neo4j_graph] Seeded GPON knowledge: {counts}")
    return {"seeded": True, **counts}


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

# Nokia GPON keyword extraction patterns
_ALARM_CODES = {"LOS", "LOF", "SF", "SD", "SUF", "LOM", "MEM", "Dying Gasp"}
_ENTITY_NAMES = {"OLT", "ONT", "ONU", "PON", "GPON", "XGS-PON", "ISAM", "SFP", "Splitter"}
_CONCEPT_TERMS = {
    "T-CONT", "TCONT", "GEM", "GEM Port", "DBA", "OMCI", "PLOAM",
    "VLAN", "QoS", "FEC", "AES", "TR-069", "Alloc-ID", "Service Port",
    "Bandwidth", "Ranging", "Activation", "Deactivation", "Upstream",
    "Downstream", "Optical Power",
}

import re as _re

def _extract_keywords(query: str) -> List[str]:
    """Extract meaningful search keywords from query."""
    keywords = set()
    query_upper = query.upper()

    # Check known terms
    for term in _CONCEPT_TERMS | _ENTITY_NAMES:
        if term.upper() in query_upper:
            keywords.add(term)

    # Fallback: words > 3 chars, excluding stop words
    _stop = {"what", "does", "this", "that", "with", "from", "have", "will",
             "about", "your", "which", "their", "there", "when", "where",
             "would", "could", "should", "they", "been", "being", "some",
             "more", "also", "into", "than", "then", "just", "only", "very",
             "explain", "describe", "tell", "configure", "show", "display"}
    if not keywords:
        for word in query.split():
            clean = _re.sub(r'[^\w-]', '', word)
            if len(clean) > 3 and clean.lower() not in _stop:
                keywords.add(clean)

    return list(keywords)[:8]


def _extract_alarm_codes(query: str) -> List[str]:
    """Extract known alarm codes from query text."""
    found = []
    query_upper = query.upper()
    for code in _ALARM_CODES:
        if code.upper() in query_upper:
            found.append(code)
    return found


def _extract_entity_names(query: str) -> List[str]:
    """Extract known entity names from query text."""
    found = []
    query_upper = query.upper()
    for name in _ENTITY_NAMES:
        if name.upper() in query_upper:
            found.append(name)
    return found


# =============================================================================
# MODULE-LEVEL CONVENIENCE
# =============================================================================

def get_connection() -> Neo4jConnection:
    """Return the singleton connection instance."""
    return Neo4jConnection.get_instance()


def init_graph() -> bool:
    """
    Full initialisation: connect → schema → seed.
    Returns True if graph is operational.
    """
    conn = get_connection()
    if not conn.connected:
        if not conn.connect():
            return False
    if not ensure_schema(conn):
        return False
    seed_gpon_knowledge(conn)
    return True

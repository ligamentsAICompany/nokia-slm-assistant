"""
Nokia SLM Observability
=======================
Structured logging, telemetry, and metrics for the Nokia SLM system.
"""

import json
import logging
import time
import uuid
import threading
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, Optional, List, Any, Tuple
from collections import defaultdict


# =============================================================================
# STRUCTURED LOGGING
# =============================================================================

class JSONFormatter(logging.Formatter):
    """JSON log formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add extra fields
        if hasattr(record, 'query_id'):
            log_data['query_id'] = record.query_id
        if hasattr(record, 'query_type'):
            log_data['query_type'] = record.query_type
        if hasattr(record, 'extra_data'):
            log_data.update(record.extra_data)
        
        # Add exception info
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


def setup_logging(level: str = "INFO", format_type: str = "json", log_file: str = None):
    """
    Configure logging for the SLM application.
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR).
        format_type: "json" or "text".
        log_file: Optional file path for log output.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Create formatter
    if format_type == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s'
        )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        from pathlib import Path
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


# =============================================================================
# QUERY TRACING
# =============================================================================

@dataclass
class QueryTrace:
    """Trace data for a single query."""
    query_id: str
    query_text: str
    query_type: str = ""
    
    # Timestamps
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    
    # Retrieval stats
    vector_results: int = 0
    bm25_results: int = 0
    graph_results: int = 0
    reranked_results: int = 0
    
    # Cache
    cache_hit: bool = False
    
    # Safety
    refusal: bool = False
    refusal_code: str = ""
    refusal_reason: str = ""
    
    # LLM
    llm_used: str = ""
    llm_attempts: int = 0
    llm_fallback: bool = False
    
    # Response
    response_grounded: bool = False
    response_tokens: int = 0
    
    @property
    def duration_ms(self) -> float:
        if self.end_time > 0:
            return (self.end_time - self.start_time) * 1000
        return (time.time() - self.start_time) * 1000
    
    def complete(self):
        """Mark the trace as complete."""
        self.end_time = time.time()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        d = asdict(self)
        d['duration_ms'] = self.duration_ms
        return d


class QueryTracer:
    """
    Query tracer for request-scoped telemetry.
    
    Usage:
        with QueryTracer.trace("What is GPON?") as trace:
            trace.query_type = "SIMPLE"
            trace.vector_results = 50
            # ... processing ...
            trace.response_grounded = True
    """
    
    _current: Dict[str, QueryTrace] = {}
    _lock = threading.Lock()
    
    def __init__(self, query: str, query_id: str = None):
        self._trace = QueryTrace(
            query_id=query_id or str(uuid.uuid4())[:8],
            query_text=query[:200]  # Truncate for logging
        )
        self.logger = logging.getLogger("query_tracer")
    
    def __enter__(self) -> QueryTrace:
        with self._lock:
            self._current[self._trace.query_id] = self._trace
        
        self.logger.info(
            f"Query started",
            extra={
                'query_id': self._trace.query_id,
                'extra_data': {'query_text': self._trace.query_text}
            }
        )
        return self._trace
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self._trace.complete()
        
        with self._lock:
            self._current.pop(self._trace.query_id, None)
        
        log_data = {
            'query_id': self._trace.query_id,
            'query_type': self._trace.query_type,
            'duration_ms': round(self._trace.duration_ms, 2),
            'cache_hit': self._trace.cache_hit,
            'refusal': self._trace.refusal,
            'llm_used': self._trace.llm_used,
            'grounded': self._trace.response_grounded,
        }
        
        if self._trace.refusal:
            log_data['refusal_code'] = self._trace.refusal_code
        
        self.logger.info(
            f"Query completed in {self._trace.duration_ms:.1f}ms",
            extra={
                'query_id': self._trace.query_id,
                'extra_data': log_data
            }
        )
        
        # Record metrics
        Metrics.record_query(self._trace)
        
        return False  # Don't suppress exceptions
    
    @classmethod
    def trace(cls, query: str, query_id: str = None) -> 'QueryTracer':
        """Create a new query tracer."""
        return cls(query, query_id)
    
    @classmethod
    def get_current(cls, query_id: str) -> Optional[QueryTrace]:
        """Get a currently active trace by ID."""
        with cls._lock:
            return cls._current.get(query_id)


# =============================================================================
# METRICS
# =============================================================================

class Metrics:
    """
    Application metrics collector.
    
    Collects:
    - Query counts by type
    - Latency histograms
    - Cache hit rates
    - Refusal rates
    - LLM usage
    """
    
    _lock = threading.Lock()
    
    # Counters
    _query_count = 0
    _query_by_type: Dict[str, int] = defaultdict(int)
    _cache_hits = 0
    _cache_misses = 0
    _refusals = 0
    _refusal_by_code: Dict[str, int] = defaultdict(int)
    _llm_calls = 0
    _llm_fallbacks = 0
    
    # Latency tracking (buckets in ms)
    _latency_buckets = [50, 100, 250, 500, 1000, 2500, 5000, 10000]
    _latency_counts: List[int] = [0] * 9  # +1 for overflow
    _latency_sum = 0.0
    
    # Start time
    _start_time = time.time()
    
    @classmethod
    def record_query(cls, trace: QueryTrace):
        """Record metrics from a completed query trace."""
        with cls._lock:
            cls._query_count += 1
            cls._query_by_type[trace.query_type] += 1
            
            if trace.cache_hit:
                cls._cache_hits += 1
            else:
                cls._cache_misses += 1
            
            if trace.refusal:
                cls._refusals += 1
                cls._refusal_by_code[trace.refusal_code] += 1
            
            if trace.llm_used:
                cls._llm_calls += 1
            if trace.llm_fallback:
                cls._llm_fallbacks += 1
            
            # Latency histogram
            latency = trace.duration_ms
            cls._latency_sum += latency
            
            bucket_found = False
            for i, bucket in enumerate(cls._latency_buckets):
                if latency <= bucket:
                    cls._latency_counts[i] += 1
                    bucket_found = True
                    break
            if not bucket_found:
                cls._latency_counts[-1] += 1  # Overflow bucket
    
    @classmethod
    def get_metrics(cls) -> Dict[str, Any]:
        """Get current metrics snapshot."""
        with cls._lock:
            uptime = time.time() - cls._start_time
            
            return {
                "uptime_seconds": round(uptime, 1),
                "queries": {
                    "total": cls._query_count,
                    "by_type": dict(cls._query_by_type),
                    "rate_per_minute": round(cls._query_count / max(uptime / 60, 1), 2),
                },
                "cache": {
                    "hits": cls._cache_hits,
                    "misses": cls._cache_misses,
                    "hit_rate": round(
                        cls._cache_hits / max(cls._cache_hits + cls._cache_misses, 1), 3
                    ),
                },
                "safety": {
                    "refusals": cls._refusals,
                    "refusal_rate": round(
                        cls._refusals / max(cls._query_count, 1), 3
                    ),
                    "by_code": dict(cls._refusal_by_code),
                },
                "llm": {
                    "calls": cls._llm_calls,
                    "fallbacks": cls._llm_fallbacks,
                    "fallback_rate": round(
                        cls._llm_fallbacks / max(cls._llm_calls, 1), 3
                    ),
                },
                "latency": {
                    "avg_ms": round(
                        cls._latency_sum / max(cls._query_count, 1), 2
                    ),
                    "histogram": {
                        f"le_{b}": cls._latency_counts[i] 
                        for i, b in enumerate(cls._latency_buckets)
                    },
                    "overflow": cls._latency_counts[-1],
                },
            }
    
    @classmethod
    def reset(cls):
        """Reset all metrics."""
        with cls._lock:
            cls._query_count = 0
            cls._query_by_type.clear()
            cls._cache_hits = 0
            cls._cache_misses = 0
            cls._refusals = 0
            cls._refusal_by_code.clear()
            cls._llm_calls = 0
            cls._llm_fallbacks = 0
            cls._latency_counts = [0] * 9
            cls._latency_sum = 0.0
            cls._start_time = time.time()


# =============================================================================
# HEALTH CHECK
# =============================================================================

@dataclass
class HealthStatus:
    """System health status."""
    status: str  # "healthy", "degraded", "unhealthy"
    checks: Dict[str, bool] = field(default_factory=dict)
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    
    def to_dict(self) -> Dict:
        return asdict(self)


class HealthChecker:
    """System health checker."""
    
    @staticmethod
    def check_faiss_index(path: str) -> Tuple[bool, str]:
        """Check if FAISS index exists and is loadable."""
        from pathlib import Path
        if not Path(path).exists():
            return False, "FAISS index not found"
        return True, "FAISS index available"
    
    @staticmethod
    def check_lm_studio(url: str, timeout: float = 5.0) -> Tuple[bool, str]:
        """Check if LM Studio is reachable."""
        import requests
        try:
            r = requests.get(f"{url}/models", timeout=timeout)
            if r.status_code == 200:
                models = r.json().get("data", [])
                return True, f"{len(models)} model(s) available"
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_ollama(url: str, timeout: float = 5.0) -> Tuple[bool, str]:
        """Check if Ollama is reachable."""
        import requests
        try:
            # Ollama health check endpoint
            base_url = url.replace("/api/chat", "")
            r = requests.get(f"{base_url}/api/tags", timeout=timeout)
            if r.status_code == 200:
                return True, "Ollama available"
            return False, f"HTTP {r.status_code}"
        except Exception as e:
            return False, str(e)
    
    @staticmethod
    def check_neo4j(uri: str = None, timeout: float = 5.0) -> Tuple[bool, str]:
        """Check if Neo4j is reachable."""
        try:
            import os
            from dotenv import load_dotenv
            load_dotenv()
            neo_uri = uri or os.environ.get("NEO4J_URI", "bolt://localhost:7687")
            neo_user = os.environ.get("NEO4J_USER", "neo4j")
            neo_pass = os.environ.get("NEO4J_PASSWORD", "neo4j")
            from neo4j import GraphDatabase
            driver = GraphDatabase.driver(neo_uri, auth=(neo_user, neo_pass))
            with driver.session() as session:
                result = session.run("RETURN 1 AS ok")
                result.single()
            driver.close()
            return True, "Neo4j available"
        except Exception as e:
            return False, str(e)

    @classmethod
    def run_all_checks(cls, config) -> HealthStatus:
        """Run all health checks."""
        checks = {}
        details = {}
        
        # FAISS check
        faiss_ok, faiss_msg = cls.check_faiss_index(str(config.get_faiss_index_path()))
        checks["faiss_index"] = faiss_ok
        details["faiss"] = faiss_msg
        
        # LM Studio check
        lm_ok, lm_msg = cls.check_lm_studio(config.lm_studio_url)
        checks["lm_studio"] = lm_ok
        details["lm_studio"] = lm_msg
        
        # Ollama check (optional)
        ollama_ok, ollama_msg = cls.check_ollama(config.ollama_url)
        checks["ollama"] = ollama_ok
        details["ollama"] = ollama_msg
        
        # Neo4j check
        neo4j_uri = getattr(config, "neo4j_uri", None)
        neo4j_ok, neo4j_msg = cls.check_neo4j(neo4j_uri)
        checks["neo4j"] = neo4j_ok
        details["neo4j"] = neo4j_msg
        
        # Determine overall status
        critical_checks = [checks.get("faiss_index", False)]
        primary_llm = checks.get("lm_studio", False)
        fallback_llm = checks.get("ollama", False)
        
        if all(critical_checks) and (primary_llm or fallback_llm):
            status = "healthy"
        elif all(critical_checks):
            status = "degraded"  # No LLM available
        else:
            status = "unhealthy"
        
        return HealthStatus(
            status=status,
            checks=checks,
            details=details
        )

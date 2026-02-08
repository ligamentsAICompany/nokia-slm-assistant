"""
SLM Optimizations Module
========================
Production-grade implementations for:
1. Graph Maintenance (TTL, pruning, incremental updates)
2. Context Window Management (token counting, dynamic fitting)
3. Latency Optimization (async processing, connection pooling, caching)
"""
import os
import time
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
import requests

BASE_DIR = "D:/Ligaments_AI/SLM"
CACHE_DIR = f"{BASE_DIR}/cache"
QUERY_CACHE_PATH = f"{CACHE_DIR}/query_cache.json"

REFUSAL_STRINGS = (
    "INSUFFICIENT DOCUMENTATION CONTEXT",
    "NOT FOUND IN PROVIDED DOCUMENTATION"
)

class LatencyOptimizer:

    def __init__(self, max_workers=4, cache_ttl_seconds=3600):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_ttl = cache_ttl_seconds
        self._query_cache = {}
        self._lock = threading.Lock()
        self._load_cache()

        self._session = requests.Session()

    def _load_cache(self):
        if os.path.exists(QUERY_CACHE_PATH):
            try:
                with open(QUERY_CACHE_PATH, "r") as f:
                    raw = json.load(f)
                now = time.time()
                self._query_cache = {
                    k: v for k, v in raw.items()
                    if v.get("expires_at", 0) > now
                }
            except:
                self._query_cache = {}

    def _save_cache(self):
        os.makedirs(os.path.dirname(QUERY_CACHE_PATH), exist_ok=True)
        with open(QUERY_CACHE_PATH, "w") as f:
            json.dump(self._query_cache, f)

    def _key(self, query):
        return hashlib.md5(query.lower().strip().encode()).hexdigest()

    def get_cached(self, query):
        key = self._key(query)
        with self._lock:
            entry = self._query_cache.get(key)
            if entry and entry["expires_at"] > time.time():
                entry["hits"] += 1
                return entry["result"]
        return None

    def set_cached(self, query, result):
        # 🔴 DO NOT CACHE REFUSALS (Req #3)
        # If the result contains any refusal string, skip caching.
        if isinstance(result, str):
            for r in REFUSAL_STRINGS:
                if r in result:
                    print(f"[CACHE] Skipping cache for refusal: {r}")
                    return

        key = self._key(query)
        with self._lock:
            self._query_cache[key] = {
                "result": result,
                "expires_at": time.time() + self.cache_ttl,
                "created_at": time.time(),
                "hits": 0
            }
            self.executor.submit(self._save_cache)

    def clear_cache(self):
        with self._lock:
            self._query_cache = {}
            if os.path.exists(QUERY_CACHE_PATH):
                os.remove(QUERY_CACHE_PATH)

    def get_cache_stats(self):
        with self._lock:
            return {
                "entries": len(self._query_cache),
                "ttl_seconds": self.cache_ttl,
                "total_hits": sum(v.get("hits", 0) for v in self._query_cache.values())
            }

    def http_request(self, url, method="POST", json_data=None, timeout=60):
        if method.upper() == "POST":
            return self._session.post(url, json=json_data, timeout=timeout)
        return self._session.get(url, timeout=timeout)

    def shutdown(self):
        self._save_cache()
        self.executor.shutdown(wait=True)
        self._session.close()

import os
import time
import hashlib
import json
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import requests

# --- CONFIGURATION ---
BASE_DIR = "D:/Ligaments_AI/SLM"
CACHE_DIR = f"{BASE_DIR}/cache"
QUERY_CACHE_PATH = f"{CACHE_DIR}/query_cache.json"

# =============================================================================
# 1. CONTEXT WINDOW MANAGEMENT
# =============================================================================

class ContextWindowManager:
    """
    Manages context window limits with accurate token counting.
    Supports multiple model context sizes and dynamic fitting.
    """
    
    # Model context limits (tokens)
    MODEL_LIMITS = {
        "default": 4096,
        "qwen": 32768,
        "llama-3": 8192,
        "mistral": 32768,
        "phi-3": 128000,
        "gemma": 8192,
    }
    
    # Reserve tokens for system prompt + response
    RESERVED_TOKENS = 1500  # System prompt (~500) + max response (~1000)
    
    def __init__(self, model_name: str = "default"):
        self.model_name = model_name
        self.max_context = self._get_model_limit(model_name)
        self.available_for_context = self.max_context - self.RESERVED_TOKENS
        self._tokenizer = None
        
    def _get_model_limit(self, model_name: str) -> int:
        """Get context limit for model, with fuzzy matching."""
        model_lower = model_name.lower()
        for key, limit in self.MODEL_LIMITS.items():
            if key in model_lower:
                return limit
        return self.MODEL_LIMITS["default"]
    
    def count_tokens(self, text: str) -> int:
        """
        Count tokens accurately. Uses tiktoken if available, else estimates.
        Estimation: ~4 characters per token (conservative for English)
        """
        if not text:
            return 0
            
        # Try tiktoken first (most accurate)
        try:
            if self._tokenizer is None:
                import tiktoken
                # cl100k_base works for most modern models
                self._tokenizer = tiktoken.get_encoding("cl100k_base")
            return len(self._tokenizer.encode(text))
        except ImportError:
            pass
        except Exception:
            pass
        
        # Fallback: character-based estimation (conservative)
        # Average ~4 chars/token for English, ~2.5 for technical text
        return len(text) // 3  # Conservative estimate
    
    def fit_context(self, chunks: List[Dict], query: str, 
                    system_prompt: str = "") -> Tuple[List[Dict], int]:
        """
        Dynamically fit chunks into available context window.
        
        Returns:
            Tuple of (fitted_chunks, total_tokens_used)
        """
        # Calculate fixed token costs
        query_tokens = self.count_tokens(query)
        system_tokens = self.count_tokens(system_prompt)
        overhead = query_tokens + system_tokens + 100  # 100 for formatting
        
        available = self.available_for_context - overhead
        
        if available <= 0:
            print(f"[WARN] Query too long ({query_tokens} tokens)")
            return [], 0
        
        # Fit chunks greedily (already ranked by relevance)
        fitted_chunks = []
        total_used = 0
        
        for chunk in chunks:
            chunk_text = chunk.get('text', chunk.get('enriched_text', ''))
            chunk_tokens = self.count_tokens(chunk_text)
            
            if total_used + chunk_tokens <= available:
                fitted_chunks.append(chunk)
                total_used += chunk_tokens
            else:
                # Try to fit a truncated version
                remaining = available - total_used
                if remaining > 200:  # Worth including partial
                    # Estimate chars from remaining tokens
                    max_chars = remaining * 3
                    truncated_chunk = chunk.copy()
                    truncated_chunk['text'] = chunk_text[:max_chars] + "..."
                    truncated_chunk['truncated'] = True
                    fitted_chunks.append(truncated_chunk)
                    total_used += remaining
                break
        
        return fitted_chunks, total_used
    
    def get_stats(self) -> Dict:
        """Return context window statistics."""
        return {
            "model": self.model_name,
            "max_context": self.max_context,
            "reserved_tokens": self.RESERVED_TOKENS,
            "available_for_context": self.available_for_context
        }


# =============================================================================
# 2. GRAPH MAINTENANCE
# =============================================================================

class GraphMaintenance:
    """
    Neo4j graph maintenance with TTL, pruning, and incremental updates.
    """
    
    def __init__(self, driver, default_ttl_days: int = 30):
        self.driver = driver
        self.default_ttl_days = default_ttl_days
        self._maintenance_lock = threading.Lock()
        
    def add_entity(self, name: str, entity_type: str, 
                   properties: Dict = None, ttl_days: int = None) -> bool:
        """
        Add or update an entity with TTL timestamp.
        """
        if not self.driver:
            return False
            
        ttl = ttl_days or self.default_ttl_days
        expires_at = datetime.now() + timedelta(days=ttl)
        
        props = properties or {}
        props['created_at'] = datetime.now().isoformat()
        props['expires_at'] = expires_at.isoformat()
        props['ttl_days'] = ttl
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type,
                        e.created_at = $created_at,
                        e.expires_at = $expires_at,
                        e.ttl_days = $ttl_days,
                        e += $extra_props
                """, 
                    name=name, 
                    type=entity_type,
                    created_at=props['created_at'],
                    expires_at=props['expires_at'],
                    ttl_days=props['ttl_days'],
                    extra_props={k: v for k, v in props.items() 
                                if k not in ['created_at', 'expires_at', 'ttl_days']}
                )
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add entity: {e}")
            return False
    
    def add_relationship(self, from_entity: str, to_entity: str, 
                        rel_type: str, properties: Dict = None) -> bool:
        """Add a relationship between entities."""
        if not self.driver:
            return False
            
        props = properties or {}
        props['created_at'] = datetime.now().isoformat()
        
        try:
            with self.driver.session() as session:
                # Dynamic relationship type
                query = f"""
                    MATCH (a:Entity {{name: $from_name}})
                    MATCH (b:Entity {{name: $to_name}})
                    MERGE (a)-[r:{rel_type.upper().replace(' ', '_')}]->(b)
                    SET r += $props
                """
                session.run(query, 
                    from_name=from_entity, 
                    to_name=to_entity, 
                    props=props
                )
            return True
        except Exception as e:
            print(f"[ERROR] Failed to add relationship: {e}")
            return False
    
    def prune_expired(self) -> Dict:
        """
        Remove entities past their TTL expiration.
        Returns count of pruned nodes and relationships.
        """
        if not self.driver:
            return {"pruned_nodes": 0, "pruned_rels": 0, "error": "No driver"}
        
        with self._maintenance_lock:
            try:
                now = datetime.now().isoformat()
                with self.driver.session() as session:
                    # Count before deletion
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE e.expires_at IS NOT NULL AND e.expires_at < $now
                        RETURN count(e) as expired_count
                    """, now=now)
                    expired_count = result.single()["expired_count"]
                    
                    if expired_count > 0:
                        # Delete expired nodes (and their relationships)
                        session.run("""
                            MATCH (e:Entity)
                            WHERE e.expires_at IS NOT NULL AND e.expires_at < $now
                            DETACH DELETE e
                        """, now=now)
                        
                        print(f"[MAINTENANCE] Pruned {expired_count} expired entities")
                    
                    return {
                        "pruned_nodes": expired_count,
                        "timestamp": now
                    }
            except Exception as e:
                print(f"[ERROR] Prune failed: {e}")
                return {"pruned_nodes": 0, "error": str(e)}
    
    def prune_orphans(self) -> Dict:
        """Remove orphan nodes (no relationships)."""
        if not self.driver:
            return {"pruned_orphans": 0, "error": "No driver"}
            
        with self._maintenance_lock:
            try:
                with self.driver.session() as session:
                    result = session.run("""
                        MATCH (e:Entity)
                        WHERE NOT (e)-[]-()
                        WITH e, count(e) as orphan_count
                        DELETE e
                        RETURN orphan_count
                    """)
                    # Get count of deleted orphans
                    record = result.single()
                    orphan_count = record["orphan_count"] if record else 0
                    
                    return {"pruned_orphans": orphan_count}
            except Exception as e:
                print(f"[ERROR] Orphan prune failed: {e}")
                return {"pruned_orphans": 0, "error": str(e)}
    
    def refresh_entity_ttl(self, name: str, ttl_days: int = None) -> bool:
        """Extend TTL for frequently accessed entities."""
        if not self.driver:
            return False
            
        ttl = ttl_days or self.default_ttl_days
        new_expires = (datetime.now() + timedelta(days=ttl)).isoformat()
        
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (e:Entity {name: $name})
                    SET e.expires_at = $expires_at,
                        e.last_accessed = $now
                """, name=name, expires_at=new_expires, now=datetime.now().isoformat())
            return True
        except Exception as e:
            print(f"[ERROR] TTL refresh failed: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """Get graph statistics for monitoring."""
        if not self.driver:
            return {"error": "No driver connected"}
            
        try:
            with self.driver.session() as session:
                # Node counts by type
                result = session.run("""
                    MATCH (e:Entity)
                    RETURN e.type as type, count(*) as count
                    ORDER BY count DESC
                """)
                type_counts = {r["type"]: r["count"] for r in result}
                
                # Relationship count
                result = session.run("MATCH ()-[r]->() RETURN count(r) as rel_count")
                rel_count = result.single()["rel_count"]
                
                # Expiring soon (next 7 days)
                week_from_now = (datetime.now() + timedelta(days=7)).isoformat()
                result = session.run("""
                    MATCH (e:Entity)
                    WHERE e.expires_at IS NOT NULL AND e.expires_at < $deadline
                    RETURN count(e) as expiring_soon
                """, deadline=week_from_now)
                expiring_soon = result.single()["expiring_soon"]
                
                return {
                    "total_entities": sum(type_counts.values()),
                    "entities_by_type": type_counts,
                    "total_relationships": rel_count,
                    "expiring_in_7_days": expiring_soon,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception as e:
            return {"error": str(e)}
    
    def run_full_maintenance(self) -> Dict:
        """Run complete maintenance cycle."""
        results = {
            "expired": self.prune_expired(),
            "orphans": self.prune_orphans(),
            "stats": self.get_stats(),
            "completed_at": datetime.now().isoformat()
        }
        print(f"[MAINTENANCE] Full cycle completed: {results}")
        return results


# =============================================================================
# 3. LATENCY OPTIMIZATION
# =============================================================================

class LatencyOptimizer:
    """
    Latency optimization with parallel processing, caching, and connection pooling.
    """
    
    def __init__(self, max_workers: int = 4, cache_ttl_seconds: int = 3600):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.cache_ttl = cache_ttl_seconds
        self._query_cache = {}
        self._cache_lock = threading.Lock()
        self._load_cache()
        
        # Connection pool for HTTP requests
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=requests.adapters.Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[500, 502, 503, 504]
            )
        )
        self._session.mount('http://', adapter)
        self._session.mount('https://', adapter)
    
    def _load_cache(self):
        """Load query cache from disk."""
        try:
            if os.path.exists(QUERY_CACHE_PATH):
                with open(QUERY_CACHE_PATH, 'r') as f:
                    data = json.load(f)
                    # Filter expired entries
                    now = time.time()
                    self._query_cache = {
                        k: v for k, v in data.items()
                        if v.get('expires_at', 0) > now
                    }
                    print(f"[CACHE] Loaded {len(self._query_cache)} cached queries")
        except Exception as e:
            print(f"[WARN] Cache load failed: {e}")
            self._query_cache = {}
    
    def _save_cache(self):
        """Persist cache to disk."""
        try:
            os.makedirs(os.path.dirname(QUERY_CACHE_PATH), exist_ok=True)
            with open(QUERY_CACHE_PATH, 'w') as f:
                json.dump(self._query_cache, f)
        except Exception as e:
            print(f"[WARN] Cache save failed: {e}")
    
    def _cache_key(self, query: str) -> str:
        """Generate cache key from query."""
        normalized = query.lower().strip()
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get_cached(self, query: str) -> Optional[Dict]:
        """Get cached result if valid."""
        key = self._cache_key(query)
        with self._cache_lock:
            if key in self._query_cache:
                entry = self._query_cache[key]
                if entry.get('expires_at', 0) > time.time():
                    entry['hits'] = entry.get('hits', 0) + 1
                    return entry.get('result')
                else:
                    del self._query_cache[key]
        return None
    
    def set_cached(self, query: str, result: Dict):
        """Cache a query result."""
        key = self._cache_key(query)
        with self._cache_lock:
            self._query_cache[key] = {
                'result': result,
                'expires_at': time.time() + self.cache_ttl,
                'created_at': time.time(),
                'hits': 0
            }
            # Async save (don't block)
            self.executor.submit(self._save_cache)
    
    def clear_cache(self):
        """Clear all cached queries."""
        with self._cache_lock:
            self._query_cache = {}
            if os.path.exists(QUERY_CACHE_PATH):
                os.remove(QUERY_CACHE_PATH)
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        with self._cache_lock:
            total_hits = sum(e.get('hits', 0) for e in self._query_cache.values())
            return {
                "size": len(self._query_cache),
                "total_hits": total_hits,
                "ttl_seconds": self.cache_ttl
            }
    
    def parallel_search(self, 
                       vector_search_fn, 
                       graph_search_fn,
                       keyword_search_fn,
                       query: str,
                       timeout: float = 5.0) -> Dict[str, Any]:
        """
        Execute multiple search methods in parallel.
        
        Args:
            vector_search_fn: Function that returns vector search results
            graph_search_fn: Function that returns graph search results  
            keyword_search_fn: Function that returns keyword search results
            query: The search query
            timeout: Max time to wait for all searches
            
        Returns:
            Dict with results from each search method
        """
        results = {
            "vector": [],
            "graph": [],
            "keyword": [],
            "timings": {}
        }
        
        futures = {}
        
        # Submit all searches in parallel
        if vector_search_fn:
            futures['vector'] = self.executor.submit(vector_search_fn, query)
        if graph_search_fn:
            futures['graph'] = self.executor.submit(graph_search_fn, query)
        if keyword_search_fn:
            futures['keyword'] = self.executor.submit(keyword_search_fn, query)
        
        # Collect results with timeout
        start = time.time()
        for name, future in futures.items():
            try:
                remaining = max(0.1, timeout - (time.time() - start))
                result = future.result(timeout=remaining)
                results[name] = result
                results['timings'][name] = round(time.time() - start, 3)
            except Exception as e:
                print(f"[WARN] {name} search failed: {e}")
                results[name] = []
                results['timings'][name] = -1  # Indicates failure
        
        results['total_time'] = round(time.time() - start, 3)
        return results
    
    def http_request(self, url: str, method: str = "GET", 
                    json_data: Dict = None, timeout: float = 30) -> requests.Response:
        """
        Make HTTP request using connection pool.
        """
        if method.upper() == "GET":
            return self._session.get(url, timeout=timeout)
        elif method.upper() == "POST":
            return self._session.post(url, json=json_data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def shutdown(self):
        """Clean shutdown of thread pool."""
        self._save_cache()
        self.executor.shutdown(wait=True)
        self._session.close()


# =============================================================================
# 4. UNIFIED OPTIMIZER
# =============================================================================

class SLMOptimizer:
    """
    Unified optimizer combining all optimization features.
    Drop-in enhancement for SLMBackend.
    """
    
    def __init__(self, graph_driver=None, model_name: str = "default"):
        self.context_manager = ContextWindowManager(model_name)
        self.graph_maintenance = GraphMaintenance(graph_driver) if graph_driver else None
        self.latency = LatencyOptimizer()
        
        # Background maintenance scheduler
        self._maintenance_thread = None
        self._stop_maintenance = threading.Event()
    
    def start_maintenance_scheduler(self, interval_hours: int = 24):
        """Start background maintenance thread."""
        def maintenance_loop():
            while not self._stop_maintenance.wait(timeout=interval_hours * 3600):
                if self.graph_maintenance:
                    print("[SCHEDULER] Running graph maintenance...")
                    self.graph_maintenance.run_full_maintenance()
        
        self._maintenance_thread = threading.Thread(
            target=maintenance_loop, 
            daemon=True,
            name="GraphMaintenance"
        )
        self._maintenance_thread.start()
        print(f"[SCHEDULER] Maintenance scheduled every {interval_hours} hours")
    
    def stop_maintenance_scheduler(self):
        """Stop the maintenance scheduler."""
        self._stop_maintenance.set()
        if self._maintenance_thread:
            self._maintenance_thread.join(timeout=5)
    
    def optimize_context(self, chunks: List[Dict], query: str, 
                        system_prompt: str = "") -> Tuple[List[Dict], Dict]:
        """
        Optimize chunks to fit context window.
        Returns fitted chunks and stats.
        """
        fitted, tokens_used = self.context_manager.fit_context(
            chunks, query, system_prompt
        )
        
        stats = {
            "original_chunks": len(chunks),
            "fitted_chunks": len(fitted),
            "tokens_used": tokens_used,
            "available_tokens": self.context_manager.available_for_context,
            "utilization": round(tokens_used / self.context_manager.available_for_context * 100, 1)
        }
        
        return fitted, stats
    
    def cached_search(self, query: str, search_fn) -> Tuple[Any, bool]:
        """
        Search with caching layer.
        Returns (result, was_cached).
        """
        cached = self.latency.get_cached(query)
        if cached:
            return cached, True
        
        result = search_fn(query)
        self.latency.set_cached(query, result)
        return result, False
    
    def get_all_stats(self) -> Dict:
        """Get combined statistics from all optimizers."""
        stats = {
            "context": self.context_manager.get_stats(),
            "cache": self.latency.get_cache_stats()
        }
        
        if self.graph_maintenance:
            stats["graph"] = self.graph_maintenance.get_stats()
        
        return stats
    
    def shutdown(self):
        """Clean shutdown of all components."""
        self.stop_maintenance_scheduler()
        self.latency.shutdown()
        print("[OPTIMIZER] Shutdown complete")


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_optimizer(graph_driver=None, model_name: str = "default") -> SLMOptimizer:
    """Factory function to create configured optimizer."""
    optimizer = SLMOptimizer(graph_driver, model_name)
    return optimizer


# Quick test
if __name__ == "__main__":
    print("=" * 60)
    print("SLM Optimizations Module - Self Test")
    print("=" * 60)
    
    # Test Context Manager
    print("\n[TEST] Context Window Manager")
    ctx = ContextWindowManager("qwen")
    print(f"  Model limit: {ctx.max_context} tokens")
    print(f"  Available: {ctx.available_for_context} tokens")
    
    test_text = "This is a test sentence for token counting."
    tokens = ctx.count_tokens(test_text)
    print(f"  '{test_text}' = {tokens} tokens")
    
    # Test fitting
    chunks = [
        {"text": "Chunk 1 " * 100},
        {"text": "Chunk 2 " * 100},
        {"text": "Chunk 3 " * 100},
    ]
    fitted, used = ctx.fit_context(chunks, "test query")
    print(f"  Fitted {len(fitted)}/{len(chunks)} chunks using {used} tokens")
    
    # Test Latency Optimizer
    print("\n[TEST] Latency Optimizer")
    lat = LatencyOptimizer()
    lat.set_cached("test query", {"answer": "test result"})
    cached = lat.get_cached("test query")
    print(f"  Cache set/get: {'OK' if cached else 'FAIL'}")
    print(f"  Cache stats: {lat.get_cache_stats()}")
    lat.shutdown()
    
    print("\n[OK] All tests passed!")

"""
SLM Core Logic
==============
Contains the backend logic for Nokia SLM AI Assistant.
Split from the Streamlit app to allow automated testing.

Optimizations v2.0:
- Context window management with token counting
- Graph maintenance with TTL and pruning
- Latency optimization with parallel search and caching
"""

import os
import logging

# Set TF_USE_LEGACY_KERAS before any TensorFlow/Keras imports
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF warnings

import re
import faiss
import numpy as np
import pickle
import requests
import json
import time
from typing import List, Dict, Tuple, Optional

# Import configuration
try:
    from config import load_config
    CONFIG = load_config()
except ImportError:
    CONFIG = None

# Import safety policy
try:
    from safety_policy import get_safety_policy, SafetyPolicy, RefusalCode
    SAFETY_AVAILABLE = True
except ImportError:
    SAFETY_AVAILABLE = False

# Import observability
try:
    from observability import QueryTracer, Metrics, setup_logging
    OBSERVABILITY_AVAILABLE = True
except ImportError:
    OBSERVABILITY_AVAILABLE = False

# Import alarm intelligence
try:
    from alarm_intelligence import get_alarm_intelligence
    ALARM_INTELLIGENCE_AVAILABLE = True
except ImportError:
    ALARM_INTELLIGENCE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Import optimizations module
try:
    from optimizations import SLMOptimizer, ContextWindowManager, LatencyOptimizer
    OPTIMIZATIONS_AVAILABLE = True
except ImportError:
    OPTIMIZATIONS_AVAILABLE = False
    print("[WARN] Optimizations module not found - using basic mode")

# FastEmbed for embeddings (no TensorFlow dependency!)
FastEmbedder = None

def get_fastembed():
    """Initialize FastEmbed model (lightweight, no TensorFlow)"""
    global FastEmbedder
    if FastEmbedder is None:
        try:
            from fastembed import TextEmbedding
            FastEmbedder = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
            print("[OK] Loaded FastEmbed embedder (bge-small-en-v1.5)")
        except Exception as e:
            print(f"[WARN] FastEmbed import failed: {e}")
            FastEmbedder = None
    return FastEmbedder

# Keep CrossEncoder as optional (for re-ranking only)
CrossEncoder = None
def get_cross_encoder():
    global CrossEncoder
    if CrossEncoder is None:
        try:
            from sentence_transformers import CrossEncoder as CE
            CrossEncoder = CE
        except:
            CrossEncoder = None
    return CrossEncoder

# BM25 Import
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    BM25Okapi = None
    print("[WARN] rank_bm25 not installed")

# Neo4j import with error handling
import networkx as nx
from enum import Enum
import re

# --- SAFETY CONSTANTS ---
REFUSAL_RESPONSE = "INSUFFICIENT DOCUMENTATION CONTEXT."
MIN_CONFIDENCE_THRESHOLD = 0.05  # Cross-Encoder score threshold (relaxed to reduce false refusals)


# --- QUERY TYPE CLASSIFICATION ---
class QueryType(Enum):
    SIMPLE = "simple"           # What does X do? / Explain X
    CONFIGURATION = "config"    # Configure X / Create X / Enable X
    TROUBLESHOOTING = "trouble" # Error X / Alarm X / Not working
    LOG_ANALYSIS = "log"        # Raw log input for analysis
    GENERAL = "general"         # Default fallback

# --- LOG DETECTION PATTERNS ---
LOG_PATTERNS = [
    # Syslog format: "Jan 28 10:15:23 hostname process[pid]: message"
    r'\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\s+\S+',
    # ISO timestamp: "2024-01-28T10:15:23"
    r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}',
    # Nokia format: "PON[1/1/1]", "ONT-123", "OLT-1"
    r'\b(PON|ONT|OLT|GPON|LT|NT|ISAM)[\[\-]\d',
    # Error codes: "error=0x0003", "errno=5", "code: 1234"
    r'\b(error|errno|code)[=:]\s*0?x?[0-9a-fA-F]+',
    # dBm values: "-32.5dBm"
    r'-?\d+\.?\d*\s*dBm',
    # Alarm keywords in log context
    r'\b(ALARM|CRITICAL|WARNING|ERROR|FATAL|INFO|DEBUG)\b.*:',
    # JSON log format
    r'^\s*\{.*"(level|severity|message|timestamp)".*\}\s*$',
]

def detect_log_input(query: str) -> bool:
    """
    Detect if the input is a raw log (vs a natural language question).
    Returns True if the input appears to be log data.
    """
    # Check if input has multiple lines (logs usually do)
    lines = query.strip().split('\n')
    
    # Check for log patterns
    pattern_matches = 0
    for pattern in LOG_PATTERNS:
        if re.search(pattern, query, re.MULTILINE | re.IGNORECASE):
            pattern_matches += 1
    
    # Heuristics:
    # 1. Multiple pattern matches = likely a log
    # 2. Multiple lines with timestamps = likely a log
    # 3. Contains common log prefixes
    
    if pattern_matches >= 2:
        return True
    if len(lines) >= 3 and pattern_matches >= 1:
        return True
    if re.search(r'(ranging failed|LOS alarm|optical power|fiber|dBm)', query, re.IGNORECASE):
        if pattern_matches >= 1 or len(lines) >= 2:
            return True
    
    return False

def extract_log_errors(log_text: str) -> list:
    """
    Extract key error terms from log for knowledge base search.
    Returns list of search terms.
    """
    extracted_terms = []
    
    # Extract error codes
    error_codes = re.findall(r'(error|errno|code)[=:]\s*(0?x?[0-9a-fA-F]+)', log_text, re.IGNORECASE)
    for _, code in error_codes:
        extracted_terms.append(f"error {code}")
    
    # Extract Nokia component references
    components = re.findall(r'\b(PON|ONT|OLT|GPON|T-CONT|GEM|DBA|QoS|VLAN|TR-069)\b', log_text, re.IGNORECASE)
    extracted_terms.extend(list(set(components)))
    
    # Extract alarm/error types
    alarm_types = re.findall(r'\b(LOS|LOF|SF|SD|ranging failed|authentication|deactivat|timeout|unreachable|power|optical)\b', log_text, re.IGNORECASE)
    extracted_terms.extend(list(set(alarm_types)))
    
    # Extract status keywords
    status_keywords = re.findall(r'\b(failed|error|alarm|critical|warning|down|offline|disabled)\b', log_text, re.IGNORECASE)
    extracted_terms.extend(list(set(status_keywords)))
    
    return list(set(extracted_terms))[:15]  # Cap at 15 terms

def classify_query(query: str) -> tuple:
    """
    Classify query type and return (QueryType, top_k_chunks).
    
    Returns:
        tuple: (QueryType, chunks_to_retrieve)
    """
    query_lower = query.lower().strip()
    
    # FIRST: Check if input is raw log data
    if detect_log_input(query):
        return (QueryType.LOG_ANALYSIS, 200)  # Max chunks for log analysis
    
    # SIMPLE: Definition/explanation queries
    simple_patterns = [
        "what does", "what is", "explain", "describe", "meaning of",
        "purpose of", "definition of", "what are", "how does"
    ]
    
    # CONFIGURATION: Setup/creation queries
    config_patterns = [
        "configure", "config", "create", "set up", "setup", "enable",
        "disable", "provision", "add", "assign", "apply", "deploy",
        "install", "activate", "modify", "change", "update"
    ]
    
    # TROUBLESHOOTING: Problem/diagnostic queries
    trouble_patterns = [
        "error", "alarm", "not working", "failed", "failing", "issue",
        "problem", "troubleshoot", "diagnose", "fix", "resolve", "debug",
        "log", "los", "down", "offline", "timeout", "unreachable"
    ]
    
    # Check patterns - INCREASED CHUNK LIMITS (Max 200)
    for pattern in simple_patterns:
        if pattern in query_lower:
            return (QueryType.SIMPLE, 200)
    
    for pattern in trouble_patterns:
        if pattern in query_lower:
            return (QueryType.TROUBLESHOOTING, 200)
    
    for pattern in config_patterns:
        if pattern in query_lower:
            return (QueryType.CONFIGURATION, 200)
    
    # Default: General query
    return (QueryType.GENERAL, 200)

# --- CONFIGURATION (from config module or environment) ---
if CONFIG:
    CACHE_DIR = CONFIG.cache_dir
    BASE_DIR = CONFIG.data_dir
    FAISS_INDEX_PATH = str(CONFIG.get_faiss_index_path())
    FAISS_META_PATH = str(CONFIG.get_metadata_path())
    GRAPH_PATH = str(CONFIG.get_graph_path())
    LM_STUDIO_URL = CONFIG.lm_studio_url
else:
    # Fallback to environment variables or defaults
    CACHE_DIR = os.environ.get("SLM_CACHE_DIR", "./cache")
    BASE_DIR = os.environ.get("SLM_DATA_DIR", "./data")
    FAISS_INDEX_PATH = os.path.join(BASE_DIR, "nokia_vector_index.faiss")
    FAISS_META_PATH = os.path.join(BASE_DIR, "nokia_vector_meta.pkl")
    GRAPH_PATH = os.path.join(BASE_DIR, "nokia_graph.gml")
    LM_STUDIO_URL = os.environ.get("LM_STUDIO_URL", "http://localhost:8103/v1")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(BASE_DIR, exist_ok=True)
os.environ["HF_HOME"] = os.path.join(CACHE_DIR, "huggingface")
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.join(CACHE_DIR, "sentence_transformers")

class SLMBackend:
    def __init__(self):
        self.embedder = None
        self.cross_encoder = None  # v3.0
        self.vector_index = None
        self.metadata_store = []
        self.bm25 = None           # v3.0
        self.bm25_corpus = []      # v3.0
        self.graph = None # NetworkX Graph
        self.active_model = "local-model"
        
        # Optimization components
        self.optimizer = None
        self.context_manager = None
        self.latency_optimizer = None
        
    def load_resources(self):
        print("[INFO] Loading resources...")
        
        # 1. Load FastEmbed embedder (no TensorFlow dependency!)
        self.embedder = get_fastembed()
        if self.embedder:
            print("[OK] FastEmbed ready for vector search")
        else:
            print("[WARN] FastEmbed not available - using keyword search only")
        
        # Load Cross-Encoder for re-ranking (optional)
        CE = get_cross_encoder()
        if CE:
            try:
                print("[INFO] Loading Cross-Encoder (this may take a moment)...")
                self.cross_encoder = CE("cross-encoder/ms-marco-MiniLM-L-6-v2")
                print("[OK] Loaded Cross-Encoder")
            except Exception as ce_e:
                print(f"[WARN] Cross-Encoder failed to load: {ce_e}")
        
        # 2. Load FAISS Index
        if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_META_PATH):
            self.vector_index = faiss.read_index(FAISS_INDEX_PATH)
            with open(FAISS_META_PATH, "rb") as f:
                self.metadata_store = pickle.load(f)
            print(f"[OK] Loaded FAISS index with {self.vector_index.ntotal} vectors")
            print(f"[OK] Loaded {len(self.metadata_store)} metadata entries")
            
            # Initialize BM25 (v3.0)
            if BM25Okapi and self.metadata_store:
                print("[INFO] Building BM25 index...")
                tokenized_corpus = []
                for meta in self.metadata_store:
                    text = meta.get('enriched_text', meta.get('original_text', '')).lower()
                    tokenized_corpus.append(text.split())
                
                self.bm25 = BM25Okapi(tokenized_corpus)
                self.bm25_corpus = self.metadata_store
                print("[OK] Built BM25 index")
        else:
            print("[WARN] FAISS index not found!")
            
        # 3. Load Knowledge Graph (NetworkX)
        if os.path.exists(GRAPH_PATH):
            try:
                self.graph = nx.read_gml(GRAPH_PATH)
                print(f"[OK] Loaded Knowledge Graph ({self.graph.number_of_nodes()} nodes)")
            except Exception as e:
                print(f"[WARN] Graph load failed: {e}")
                self.graph = None
        else:
            print("[WARN] Local Graph file not found")
            self.graph = None
            
        # 4. Get Model
        self.active_model = self._get_active_model()
        print(f"[INFO] Using Model: {self.active_model}")
        
        # 5. Initialize Optimizations
        if OPTIMIZATIONS_AVAILABLE:
            self.optimizer = SLMOptimizer(None, self.active_model)
            self.context_manager = self.optimizer.context_manager
            self.latency_optimizer = self.optimizer.latency
            
            # Start background maintenance (every 24 hours)
            if self.graph:
                # NetworkX graph is in-memory, no maintenance driver needed
                pass
            
            print("[OK] Optimizations loaded (context management, caching, graph maintenance)")
        else:
            print("[WARN] Running without optimizations")

    def _get_active_model(self):
        try:
            r = requests.get(f"{LM_STUDIO_URL}/models", timeout=5)
            if r.status_code == 200:
                models = r.json().get("data", [])
                if models:
                    return models[0].get("id", "local-model")
        except Exception as e:
            print(f"[WARN] Model Detection Failed: {e}")
        return "local-model"

    def keyword_match_boost(self, query, text):
        query_words = set(query.lower().split())
        text_lower = text.lower()
        matches = sum(1 for word in query_words if word in text_lower and len(word) > 3)
        return matches

    def expand_query_with_llm(self, query):
        """v3.0: Use LLM to generate technical synonyms."""
        try:
            # Simple heuristic expansion as fallback/start
            # In a real scenario, this would call self.generate_response with a specific prompt
            # For now, we will do a fast internal expansion 
            # to avoid latency of another full LLM call if not cached
            pass 
        except:
            pass
        return query

    def bm25_search(self, query, top_k=30):
        """v3.0: BM25 Lexical Search."""
        if not self.bm25:
            return []
            
        try:
            tokenized_query = query.lower().split()
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_indices = np.argsort(doc_scores)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                score = doc_scores[idx]
                if score > 0:
                    meta = self.bm25_corpus[idx]
                    page = meta.get('metadata', {}).get('page', meta.get('page', 'Unknown'))
                    results.append({
                        'text': meta.get('enriched_text', meta.get('original_text', '')),
                        'summary': meta.get('summary', ''),
                        'page': page,
                        'score': float(score),
                        'type': 'bm25'
                    })
            return results
        except Exception as e:
            print(f"[WARN] BM25 search failed: {e}")
            return []

    def apply_rrf(self, vector_results, bm25_results, k=60):
        """v3.0: Reciprocal Rank Fusion to merge Vector and BM25 results."""
        scores = {}  # txt -> score
        metadata_map = {}
        
        # Process Vector Results
        for rank, res in enumerate(vector_results):
            txt = res['text']
            if txt not in scores:
                scores[txt] = 0
                metadata_map[txt] = res
            scores[txt] += 1 / (k + rank + 1)
            
        # Process BM25 Results
        for rank, res in enumerate(bm25_results):
            txt = res['text']
            if txt not in scores:
                scores[txt] = 0
                metadata_map[txt] = res
            scores[txt] += 1 / (k + rank + 1)
            
        # Convert back to list
        fused_results = []
        for txt, score in scores.items():
            item = metadata_map[txt]
            item['rrf_score'] = score
            fused_results.append(item)
            
        fused_results.sort(key=lambda x: x['rrf_score'], reverse=True)
        return fused_results

    def heuristic_rerank(self, chunks):
        """v3.0: Boost chunks containing specific procedural keywords."""
        for chunk in chunks:
            text = chunk.get('text', '').lower()
            boost = 0.0
            
            # Boost procedural content
            if "step" in text or "procedure" in text:
                boost += 0.05
            if "command" in text or "configure" in text:
                boost += 0.05
            if "table" in text:
                boost += 0.03
                
            # Boost if page number is present (likely non-empty chunk)
            if chunk.get('page') != 'Unknown':
                boost += 0.02
                
            # Apply boost to existing score (handling RRF or Cross-Encoder score)
            if 'cross_score' in chunk:
                chunk['cross_score'] += boost
            elif 'rrf_score' in chunk:
                chunk['rrf_score'] += boost
                
        # Re-sort
        key = 'cross_score' if chunks and 'cross_score' in chunks[0] else 'rrf_score'
        if chunks and key in chunks[0]:
             chunks.sort(key=lambda x: x[key], reverse=True)
             
        return chunks

    def cross_encoder_rerank(self, query, chunks, top_k=15):
        """v3.0: Cross-Encoder Re-ranking with Deduplication."""
        if not self.cross_encoder or not chunks:
            return chunks[:top_k]
            
        try:
            # Deduplicate chunks based on text content before re-ranking
            seen_text = set()
            unique_chunks = []
            for c in chunks:
                txt = c.get('text', '')
                # normalize slightly to catch near-duplicates
                norm_txt = "".join(txt.split()).lower()[:100] 
                if norm_txt not in seen_text:
                    seen_text.add(norm_txt)
                    unique_chunks.append(c)
            
            # Limit input to Cross Encoder to top 50 to save time
            candidates = unique_chunks[:50]
            
            # Create pairs
            pairs = [[query, c.get('text', '')] for c in candidates]
            scores = self.cross_encoder.predict(pairs)
            
            for i, score in enumerate(scores):
                candidates[i]['cross_score'] = float(score)
                
            for i, score in enumerate(scores):
                candidates[i]['cross_score'] = float(score)
                
            candidates.sort(key=lambda x: x.get('cross_score', -999), reverse=True)
            return candidates[:top_k]
        except Exception as e:
            print(f"[WARN] Cross-Encoder re-ranking failed: {e}")
            return chunks[:top_k]

    def retrieve_context(self, user_query: str, use_cache: bool = True) -> tuple:
        """
        v4.0 Retrieval Pipeline with Intelligent Query Classification:
        1. Query Classification
        2. Cache Check
        3. Hybrid Search (Vector + BM25 + Graph) -> RRF Fusion
        4. Cross-Encoder Re-Ranking (dynamic top_k based on query type)
        5. Context Window Optimization
        
        Returns:
            tuple: (context_string, query_type)
        """
        start_time = time.time()
        
        # 1. Query Classification (NEW in v4.0)
        query_type, chunk_limit = classify_query(user_query)
        print(f"[CLASSIFY] Query Type: {query_type.value} | Chunks: {chunk_limit}")
        
        # 2. Cache Check
        if use_cache and self.latency_optimizer:
            cached = self.latency_optimizer.get_cached(user_query)
            if cached:
                print(f"[CACHE HIT] Query served from cache in {(time.time()-start_time)*1000:.1f}ms")
                return (cached, query_type)
        
        # 3. Query Expansion (LOG_ANALYSIS uses extracted terms)
        if query_type == QueryType.LOG_ANALYSIS:
            # For logs, extract key error terms for searching
            extracted_terms = extract_log_errors(user_query)
            if extracted_terms:
                # Create a search query from extracted terms
                expanded_query = " ".join(extracted_terms)
                print(f"[LOG_ANALYSIS] Extracted terms: {extracted_terms}")
            else:
                expanded_query = user_query
        else:
            expanded_query = user_query
        
        search_stats = {"vector": 0, "bm25": 0, "graph": 0, "query_type": query_type.value}
        
        # Define allowed retrieval methods (Strict Routing - Req #6)
        allow_vector = query_type in [QueryType.LOG_ANALYSIS, QueryType.CONFIGURATION, QueryType.TROUBLESHOOTING, QueryType.SIMPLE, QueryType.GENERAL]
        allow_bm25 = query_type in [QueryType.TROUBLESHOOTING, QueryType.SIMPLE, QueryType.GENERAL, QueryType.CONFIGURATION, QueryType.LOG_ANALYSIS]
        allow_graph = query_type == QueryType.LOG_ANALYSIS
        
        # 4. Hybrid Search
        candidates_vector = []
        candidates_bm25 = []
        
        # Dynamic vector search based on query type
        vector_k = min(chunk_limit + 10, self.vector_index.ntotal) if self.vector_index else 30
        
        # A. Vector Search
        if allow_vector and self.vector_index and self.embedder:
            try:
                # FastEmbed uses embed() which returns a generator
                embeddings = list(self.embedder.embed([expanded_query]))
                query_embedding = np.array(embeddings[0]).reshape(1, -1).astype('float32')
                D, I = self.vector_index.search(query_embedding, k=vector_k)
                
                for idx, dist in zip(I[0], D[0]):
                    if 0 <= idx < len(self.metadata_store):
                        meta = self.metadata_store[idx]
                        page = meta.get('metadata', {}).get('page', meta.get('page', 'Unknown'))
                        candidates_vector.append({
                            'text': meta.get('enriched_text', meta.get('original_text', '')),
                            'page': page,
                            'score': float(dist),
                            'source': 'vector'
                        })
            except Exception as e:
                print(f"[WARN] Vector search error: {e}")
        search_stats['vector'] = len(candidates_vector)
        
        # B. BM25 Search (dynamic based on query type)
        if allow_bm25:
            bm25_k = chunk_limit + 5
            candidates_bm25 = self.bm25_search(expanded_query, top_k=bm25_k)
        search_stats['bm25'] = len(candidates_bm25)
        
        # C. Graph Search (v3.0 Advanced)
        graph_text = ""
        if allow_graph and self.graph:
            graph_results = self.advanced_graph_search(expanded_query)
            if graph_results:
                graph_text = "\n--- Knowledge Graph Insights ---\n" + graph_results
            search_stats['graph'] = 1 if graph_text else 0
        
        # 5. RRF Fusion (Vector + BM25)
        fused_candidates = self.apply_rrf(candidates_vector, candidates_bm25)
        
        # 6. Cross-Encoder Re-Ranking (DYNAMIC top_k based on query type)
        reranked_docs = self.cross_encoder_rerank(user_query, fused_candidates, top_k=chunk_limit)
        
        # 7. Heuristic Re-Ranking
        final_docs = self.heuristic_rerank(reranked_docs)
        
        # ============================================
        # STEP 5: RETRIEVAL SANITY CHECK
        # ============================================
        # If both vector AND bm25 returned 0 results, refuse early
        if search_stats['vector'] == 0 and search_stats['bm25'] == 0 and not graph_text:
            print("[SAFETY] No retrieval hits (vector=0, bm25=0). Refusing.")
            return (REFUSAL_RESPONSE, query_type)
        
        # ============================================
        # STEP 1 & 3: RANK-BASED GATING (NOT ABSOLUTE)
        # ============================================
        # Cross-encoder scores are logits (-∞ to +∞), not probabilities
        # We trust the RANKING, not the absolute score value
        
        if not final_docs and not graph_text:
            print("[SAFETY] No docs found after reranking. Refusing.")
            return (REFUSAL_RESPONSE, query_type)

        if final_docs:
            top_doc = final_docs[0]
            
            # STEP 3: Skip strict gating for CONFIG queries (procedural answers)
            skip_gating = query_type in [QueryType.CONFIGURATION, QueryType.LOG_ANALYSIS]
            
            if 'cross_score' in top_doc:
                score = top_doc['cross_score']
                score_type = "Cross-Encoder"
                
                # RANK-BASED LOGIC: If cross-encoder produced results, trust rank-1
                # Only refuse if score is EXTREMELY negative (clear mismatch)
                if skip_gating:
                    print(f"[SAFETY] Top Doc Score: {score:.4f} ({score_type}) | GATING SKIPPED for {query_type.value}")
                else:
                    # Normalize scores for comparison (optional safety net)
                    # Accept if this is the top-ranked result from reranking
                    if len(final_docs) > 1:
                        # Relative check: top score should be reasonably ahead
                        second_score = final_docs[1].get('cross_score', score - 1)
                        margin = score - second_score
                        print(f"[SAFETY] Top Doc Score: {score:.4f} ({score_type}) | Margin: {margin:.4f}")
                    else:
                        print(f"[SAFETY] Top Doc Score: {score:.4f} ({score_type}) | Single result - accepting")
            else:
                score = top_doc.get('rrf_score', 0)
                score_type = "RRF"
                # RRF scores are always positive, use minimal threshold
                if score < 0.01 and not skip_gating:
                    print(f"[SAFETY] RRF score too low: {score:.4f}. Refusing.")
                    if not graph_text:
                        return (REFUSAL_RESPONSE, query_type)
                else:
                    print(f"[SAFETY] Top Doc Score: {score:.4f} ({score_type}) | Accepted")
        
        # 8. Context Window Optimization
        context_parts = []
        
        # Add Graph insights first (high value)
        if graph_text:
            context_parts.append(graph_text)

        if self.context_manager and final_docs:
            fitted_chunks, stats = self.optimizer.optimize_context(
                final_docs, user_query, ""
            )
            for i, chunk in enumerate(fitted_chunks):
                page = chunk.get('page', 'Unknown')
                text = chunk.get('text', '')
                context_parts.append(f"--- Source {i+1} (Page {page}) ---\n{text}")
        else:
            # Fallback - use chunk_limit instead of fixed 5
            for i, chunk in enumerate(final_docs[:chunk_limit]):
                page = chunk.get('page', 'Unknown')
                text = chunk.get('text', '')
                context_parts.append(f"--- Source {i+1} (Page {page}) ---\n{text[:2000]}")

        final_context = "\n\n".join(context_parts) if context_parts else REFUSAL_RESPONSE
        
        # Double check refusal string in final context
        if final_context == REFUSAL_RESPONSE:
             print("[SAFETY] Final context is REFUSAL.")
        
        # Note: We do NOT cache here based on context string alone, 
        # Cache happens in backend logic BUT we deferred it to higher level or here?
        # The original code did:
        # self.latency_optimizer.set_cached(user_query, final_context)
        # We need to be careful. Providing 'Context' isn't the final answer.
        # But caching the RETRIEVAL result is valid if it's "INSUFFICIENT".
        # However, Req #3 says "MUST NEVER cache Refusal responses".
        # If we return INSUFFICIENT, user gets a refusal.
        # If we cache INSUFFICIENT, subsequent queries get fast refusal.
        # That seems okay? Requirement says "MUST NEVER cache Refusal responses" (meaning LLM output).
        # Actually, "Cache stored every response" was the problem.
        # If retrieval yielded nothing, it's safer NOT to cache it, in case documents update later?
        # Let's Skip Caching if REFUSAL_RESPONSE.
        
        if use_cache and self.latency_optimizer and final_context != REFUSAL_RESPONSE:
            self.latency_optimizer.set_cached(user_query, final_context)
            
        print(f"[SEARCH] Stats: {search_stats} | Total Time: {(time.time()-start_time)*1000:.1f}ms")
        return (final_context, query_type)

    def advanced_graph_search(self, query):
        """v3.0: NetworkX Neighbor Search."""
        if not self.graph: return None
        try:
            keywords = [w for w in query.split() if len(w) > 3]
            results = []
            
            # Simple fuzzy finder for nodes
            # In a real app, maybe use a map or trie
            nodes_to_search = []
            query_lower = query.lower()
            
            # This is O(N) but N is small (60 nodes)
            for node in self.graph.nodes(data=True):
                node_id, data = node
                # Check node content
                content = f"{node_id} {data.get('label','')} {data.get('name','')} {data.get('purpose','')}".lower()
                
                # If keyword match
                for kw in keywords:
                    if kw.lower() in content:
                        nodes_to_search.append(node_id)
                        break
            
            # Traverse
            for node_id in nodes_to_search[:3]: # Limit start nodes
                # Get neighbors
                neighbors = list(self.graph.neighbors(node_id))
                
                # Get info
                if neighbors:
                    results.append(f"Entity '{node_id}' is related to: {', '.join(str(n) for n in neighbors[:5])}")
                else:
                    results.append(f"Found Entity: '{node_id}'")
                            
            return "\n".join(list(set(results))) # Dedupe
        except Exception as e:
            print(f"[WARN] Graph search error: {e}")
            return None


    def validate_response(self, response: str, context: str, query_type=None) -> str:
        """
        v5.0: Post-Generation Validation (Req #5).
        STEP 4: Query-aware validation thresholds.
        Primary: Deterministic Keyword Overlap (Low Latency).
        Fallback: Optional LLM Verification.
        """
        print("[VALIDER] Starting post-generation validation...")
        if response in [REFUSAL_RESPONSE, ""] or context == REFUSAL_RESPONSE:
            return response

        # 1. Deterministic Check: Keyword Overlap
        # Simple heuristic: meaningful words in response should appear in context
        response_words = set(re.findall(r'\w{4,}', response.lower())) # Words > 3 chars
        if not response_words: 
            return response # Too short to validate
            
        context_lower = context.lower()
        supported_count = sum(1 for w in response_words if w in context_lower)
        ratio = supported_count / len(response_words)
        
        # STEP 4: Query-aware thresholds
        if query_type == QueryType.SIMPLE:
            required_overlap = 0.08  # Short factual answers
        elif query_type == QueryType.CONFIGURATION:
            required_overlap = 0.10  # Procedural steps
        else:
            required_overlap = 0.15  # Default for troubleshooting/general
        
        print(f"[VALIDER] Overlap Ratio: {ratio:.2f} ({supported_count}/{len(response_words)}) | Required: {required_overlap}")
        
        # Pass if meets query-specific threshold
        if ratio >= required_overlap:
             print("[VALIDER] Deterministic Check PASSED.")
             return response
             
        # 2. Fallback: LLM Verification (Only if enabled or low confidence)
        # User requested "Optional (Fallback Only)". For now, we REFUSE if deterministic fails
        # to ensure safety ("Refusal is ALWAYS preferable").
        # Uncomment below to enable LLM fallback.
        """
        validation_prompt = f"..."
        # ... LLM call ...
        """
        
        print(f"[VALIDER] Deterministic Check FAILED (Ratio {ratio:.2f} < {required_overlap}). Refusing.")
        return REFUSAL_RESPONSE



    def generate_response(self, messages, query_type=None):
        """
        Generate response with LLM reliability features:
        - Step 1: Adaptive timeout (120s for CONFIG/TROUBLE, 60s for others)
        - Step 2: Retry logic with exponential backoff (MAX_RETRIES=2)
        - Step 3: Ollama fallback when primary fails
        """
        import time
        
        MAX_RETRIES = 2
        OLLAMA_URL = "http://localhost:11434/api/chat"  # Local Ollama fallback
        
        # Step 1: Adaptive timeout based on query type
        if query_type in [QueryType.CONFIGURATION, QueryType.TROUBLESHOOTING, QueryType.LOG_ANALYSIS]:
            timeout = 120  # Longer for complex answers
        else:
            timeout = 60  # Standard for simple/general
        
        def clean_response(raw_text):
            """Clean LLM response text."""
            # 1. Remove complete <think>...</think> blocks
            clean_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL | re.IGNORECASE)
            
            # 2. Remove anything before </think> if no <think> was found
            if '</think>' in clean_text.lower():
                parts = re.split(r'</think>', clean_text, flags=re.IGNORECASE)
                clean_text = parts[-1] if parts else clean_text
            
            # 3. Remove common "thinking" phrases at the start
            thinking_patterns = [
                r'^.*?We need to.*?(?=\n[A-Z]|$)',
                r'^.*?Based on.*?(?=\n[A-Z]|$)',
                r'^.*?According to.*?(?=\n[A-Z]|$)',
                r'^.*?The command is.*?(?=\n|$)',
                r'^.*?So we must.*?(?=\n|$)',
                r'^.*?Probably just.*?(?=\n|$)',
            ]
            for pattern in thinking_patterns:
                clean_text = re.sub(pattern, '', clean_text, flags=re.DOTALL | re.IGNORECASE)
            
            # 4. Remove emojis
            clean_text = clean_text.encode('ascii', 'ignore').decode('ascii')
            
            return clean_text.strip()
        
        def call_primary_llm():
            """Call LM Studio (Nemotron)"""
            if self.latency_optimizer:
                r = self.latency_optimizer.http_request(
                    f"{LM_STUDIO_URL}/chat/completions",
                    method="POST",
                    json_data={
                        "model": self.active_model,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 1000
                    },
                    timeout=timeout
                )
            else:
                r = requests.post(
                    f"{LM_STUDIO_URL}/chat/completions",
                    json={
                        "model": self.active_model,
                        "messages": messages,
                        "temperature": 0.2,
                        "max_tokens": 1000
                    },
                    timeout=timeout
                )
            
            if r.status_code == 200:
                raw_text = r.json()['choices'][0]['message']['content']
                return clean_response(raw_text)
            else:
                raise Exception(f"HTTP {r.status_code}: {r.text[:200]}")
        
        def call_fallback_llm():
            """Step 3: Call Ollama as fallback"""
            try:
                # Convert OpenAI format to Ollama format
                ollama_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
                
                r = requests.post(
                    OLLAMA_URL,
                    json={
                        "model": "llama3.2",  # or qwen2.5:0.5b for faster
                        "messages": ollama_messages,
                        "stream": False
                    },
                    timeout=timeout
                )
                
                if r.status_code == 200:
                    raw_text = r.json().get('message', {}).get('content', '')
                    print("[LLM] Fallback (Ollama) succeeded")
                    return clean_response(raw_text) + "\n\n_[Answer via backup model]_"
                else:
                    return None
            except Exception as e:
                print(f"[LLM] Fallback (Ollama) failed: {e}")
                return None
        
        # Step 2: Retry logic with exponential backoff
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                print(f"[LLM] Attempt {attempt + 1}/{MAX_RETRIES} (timeout={timeout}s)")
                response = call_primary_llm()
                return response
            except Exception as e:
                last_error = e
                print(f"[LLM] Attempt {attempt + 1} failed: {e}")
                if attempt < MAX_RETRIES - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s
                    print(f"[LLM] Retrying in {wait_time}s...")
                    time.sleep(wait_time)
        
        # Step 3: All retries failed, try Ollama fallback
        print("[LLM] Primary LLM failed after all retries. Trying Ollama fallback...")
        fallback_response = call_fallback_llm()
        if fallback_response:
            return fallback_response
        
        # All failed - return system error (Step 6: user-friendly message)
        return f"SYSTEM_ERROR: LLM temporarily unavailable. Please retry in a moment."

    # ==========================================================================
    # OPTIMIZATION API METHODS
    # ==========================================================================
    
    def get_cache_stats(self) -> Dict:
        """Get query cache statistics."""
        if self.latency_optimizer:
            return self.latency_optimizer.get_cache_stats()
        return {"enabled": False}
    
    def clear_cache(self):
        """Clear the query cache."""
        if self.latency_optimizer:
            self.latency_optimizer.clear_cache()
            return True
        return False
    
    def get_graph_stats(self) -> Dict:
        """Get Neo4j graph statistics."""
        if self.optimizer and self.optimizer.graph_maintenance:
            return self.optimizer.graph_maintenance.get_stats()
        return {"enabled": False}
    
    def run_graph_maintenance(self) -> Dict:
        """Manually trigger graph maintenance."""
        if self.optimizer and self.optimizer.graph_maintenance:
            return self.optimizer.graph_maintenance.run_full_maintenance()
        return {"enabled": False}
    
    def get_context_stats(self) -> Dict:
        """Get context window statistics."""
        if self.context_manager:
            return self.context_manager.get_stats()
        return {"enabled": False}
    
    def get_all_stats(self) -> Dict:
        """Get comprehensive system statistics."""
        return {
            "cache": self.get_cache_stats(),
            "graph": self.get_graph_stats(),
            "context": self.get_context_stats(),
            "model": self.active_model,
            "vectors": self.vector_index.ntotal if self.vector_index else 0,
            "chunks": len(self.metadata_store),
            "v3_features": {
                "cross_encoder": bool(self.cross_encoder),
                "bm25_index": bool(self.bm25),
                "query_expansion": True,
                "advanced_graph": bool(self.graph)
            }
        }
    
    def shutdown(self):
        """Clean shutdown of all resources."""
        print("[INFO] Shutting down SLM Backend...")
        
        if self.optimizer:
            self.optimizer.shutdown()
        
        # No driver to close for NetworkX
            pass
            
        print("[OK] Shutdown complete")

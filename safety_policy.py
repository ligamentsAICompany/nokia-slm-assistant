"""
Nokia SLM Safety Policy
=======================
Centralized safety rules for the Nokia SLM system.
All safety checks, refusal logic, and validation are defined here.
"""

import re
import logging
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Set

logger = logging.getLogger(__name__)


class RefusalCode(Enum):
    """Enumeration of refusal reason codes."""
    NONE = "NONE"
    EMPTY_QUERY = "R001_EMPTY_QUERY"
    QUERY_TOO_SHORT = "R002_QUERY_TOO_SHORT"
    QUERY_TOO_LONG = "R003_QUERY_TOO_LONG"
    INJECTION_DETECTED = "R004_INJECTION_DETECTED"
    NO_RETRIEVAL_RESULTS = "R005_NO_RETRIEVAL_RESULTS"
    LOW_RELEVANCE_SCORE = "R006_LOW_RELEVANCE_SCORE"
    VALIDATION_FAILED = "R007_VALIDATION_FAILED"
    OUT_OF_SCOPE = "R008_OUT_OF_SCOPE"
    SYSTEM_ERROR = "R009_SYSTEM_ERROR"


@dataclass
class SafetyResult:
    """Result of a safety check."""
    passed: bool
    code: RefusalCode = RefusalCode.NONE
    reason: str = ""
    details: dict = field(default_factory=dict)
    
    def __bool__(self) -> bool:
        return self.passed


class SafetyPolicy:
    """
    Centralized safety policy for the Nokia SLM.
    
    Implements:
    - Pre-retrieval checks
    - Post-retrieval sanity checks
    - Post-generation validation
    - Declarative rule definitions
    """
    
    # === Safety Constants ===
    REFUSAL_RESPONSE = "INSUFFICIENT DOCUMENTATION CONTEXT."
    
    # === Query Constraints ===
    MIN_QUERY_LENGTH = 3
    MAX_QUERY_LENGTH = 10000
    
    # === Injection Patterns ===
    INJECTION_PATTERNS = [
        r"ignore\s+(all\s+)?previous\s+instructions?",
        r"ignore\s+(all\s+)?above\s+instructions?",
        r"disregard\s+(all\s+)?previous",
        r"forget\s+(all\s+)?previous",
        r"you\s+are\s+now\s+a",
        r"act\s+as\s+a",
        r"pretend\s+(you\s+are|to\s+be)",
        r"jailbreak",
        r"dan\s+mode",
        r"developer\s+mode",
        r"bypass\s+restrictions?",
        r"override\s+safety",
        r"system\s*:\s*",
        r"\[system\]",
        r"<\s*system\s*>",
    ]
    
    # === Out-of-Scope Patterns ===
    OUT_OF_SCOPE_PATTERNS = [
        r"\b(joke|funny|humor)\b",
        r"\b(weather|sports|news)\b",
        r"\b(recipe|cooking|food)\b",
        r"\b(movie|music|entertainment)\b",
        r"\b(stock|bitcoin|crypto)\b",
        r"\b(politics|election|president)\b",
    ]
    
    # === Thinking Patterns to Remove ===
    THINKING_PATTERNS = [
        r"<think>.*?</think>",
        r"<thinking>.*?</thinking>",
        r"<reasoning>.*?</reasoning>",
        r"^.*?We need to.*?(?=\n[A-Z]|$)",
        r"^.*?Let me.*?(?=\n[A-Z]|$)",
        r"^.*?I will.*?(?=\n[A-Z]|$)",
        r"^.*?First,.*?(?=\n[A-Z]|$)",
    ]
    
    # === Refusal Strings (for cache exclusion) ===
    REFUSAL_STRINGS = (
        "INSUFFICIENT DOCUMENTATION CONTEXT",
        "NOT FOUND IN PROVIDED DOCUMENTATION",
        "I cannot find",
        "no relevant information",
    )
    
    def __init__(self, config=None):
        """Initialize safety policy with optional config."""
        if config:
            self.min_confidence = config.min_confidence_threshold
            self.overlap_simple = config.validation_overlap_simple
            self.overlap_config = config.validation_overlap_config
            self.overlap_default = config.validation_overlap_default
        else:
            self.min_confidence = 0.05
            self.overlap_simple = 0.08
            self.overlap_config = 0.10
            self.overlap_default = 0.15
        
        # Compile regex patterns
        self._injection_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) 
            for p in self.INJECTION_PATTERNS
        ]
        self._out_of_scope_patterns = [
            re.compile(p, re.IGNORECASE) 
            for p in self.OUT_OF_SCOPE_PATTERNS
        ]
        self._thinking_patterns = [
            re.compile(p, re.IGNORECASE | re.DOTALL) 
            for p in self.THINKING_PATTERNS
        ]
    
    # =========================================================================
    # PRE-RETRIEVAL CHECKS
    # =========================================================================
    
    def check_query_pre_retrieval(self, query: str) -> SafetyResult:
        """
        Perform pre-retrieval safety checks on the query.
        
        Checks:
        - Empty query
        - Query length bounds
        - Injection attempts
        - Out-of-scope topics
        """
        # Empty check
        if not query or not query.strip():
            return SafetyResult(
                passed=False,
                code=RefusalCode.EMPTY_QUERY,
                reason="Query is empty."
            )
        
        query_stripped = query.strip()
        
        # Length checks
        if len(query_stripped) < self.MIN_QUERY_LENGTH:
            return SafetyResult(
                passed=False,
                code=RefusalCode.QUERY_TOO_SHORT,
                reason=f"Query too short (min {self.MIN_QUERY_LENGTH} chars)."
            )
        
        if len(query_stripped) > self.MAX_QUERY_LENGTH:
            return SafetyResult(
                passed=False,
                code=RefusalCode.QUERY_TOO_LONG,
                reason=f"Query too long (max {self.MAX_QUERY_LENGTH} chars)."
            )
        
        # Injection check
        for pattern in self._injection_patterns:
            if pattern.search(query_stripped):
                logger.warning(f"[SAFETY] Injection pattern detected: {pattern.pattern}")
                return SafetyResult(
                    passed=False,
                    code=RefusalCode.INJECTION_DETECTED,
                    reason="Query contains disallowed patterns.",
                    details={"pattern": pattern.pattern}
                )
        
        # Out-of-scope check (soft)
        for pattern in self._out_of_scope_patterns:
            if pattern.search(query_stripped):
                logger.info(f"[SAFETY] Out-of-scope topic detected: {pattern.pattern}")
                return SafetyResult(
                    passed=False,
                    code=RefusalCode.OUT_OF_SCOPE,
                    reason="Query appears to be outside Nokia documentation scope.",
                    details={"pattern": pattern.pattern}
                )
        
        return SafetyResult(passed=True)
    
    # =========================================================================
    # POST-RETRIEVAL CHECKS
    # =========================================================================
    
    def check_retrieval_sanity(
        self, 
        vector_count: int, 
        bm25_count: int, 
        graph_hits: bool = False
    ) -> SafetyResult:
        """
        Check if retrieval returned any results.
        
        Args:
            vector_count: Number of vector search results.
            bm25_count: Number of BM25 search results.
            graph_hits: Whether graph search returned results.
        """
        if vector_count == 0 and bm25_count == 0 and not graph_hits:
            return SafetyResult(
                passed=False,
                code=RefusalCode.NO_RETRIEVAL_RESULTS,
                reason="No retrieval results from any source.",
                details={
                    "vector_count": vector_count,
                    "bm25_count": bm25_count,
                    "graph_hits": graph_hits
                }
            )
        return SafetyResult(passed=True)
    
    def check_relevance_score(
        self, 
        top_score: float, 
        score_type: str, 
        query_type: str
    ) -> SafetyResult:
        """
        Check if the top document has sufficient relevance.
        
        For configuration and log queries, gating is relaxed.
        """
        # Skip gating for procedural queries
        if query_type in ["config", "log", "LOG_ANALYSIS", "CONFIGURATION"]:
            logger.debug(f"[SAFETY] Gating skipped for query_type={query_type}")
            return SafetyResult(passed=True)
        
        # RRF scores are always positive
        if score_type == "rrf" and top_score < 0.01:
            return SafetyResult(
                passed=False,
                code=RefusalCode.LOW_RELEVANCE_SCORE,
                reason=f"RRF score too low: {top_score:.4f}",
                details={"score": top_score, "type": score_type}
            )
        
        # Cross-encoder scores are logits, trust ranking over absolute value
        # Only refuse if extremely negative
        if score_type == "cross_encoder" and top_score < -10.0:
            return SafetyResult(
                passed=False,
                code=RefusalCode.LOW_RELEVANCE_SCORE,
                reason=f"Cross-encoder score extremely low: {top_score:.4f}",
                details={"score": top_score, "type": score_type}
            )
        
        return SafetyResult(passed=True)
    
    # =========================================================================
    # POST-GENERATION VALIDATION
    # =========================================================================
    
    def validate_response(
        self, 
        response: str, 
        context: str, 
        query_type: str = None
    ) -> Tuple[str, SafetyResult]:
        """
        Validate LLM response against context.
        
        Uses keyword overlap to detect hallucination.
        
        Args:
            response: The LLM-generated response.
            context: The source context used for generation.
            query_type: The classified query type.
            
        Returns:
            Tuple of (validated_response, safety_result)
        """
        # Already a refusal
        if self.is_refusal(response):
            return response, SafetyResult(passed=True)
        
        if self.is_refusal(context):
            return response, SafetyResult(passed=True)
        
        # Clean response of thinking patterns
        cleaned = self.clean_response(response)
        
        if not cleaned.strip():
            return self.REFUSAL_RESPONSE, SafetyResult(
                passed=False,
                code=RefusalCode.VALIDATION_FAILED,
                reason="Response empty after cleaning."
            )
        
        # Calculate keyword overlap
        response_words = set(re.findall(r'\w{4,}', cleaned.lower()))
        if not response_words:
            return cleaned, SafetyResult(passed=True)
        
        context_lower = context.lower()
        supported_count = sum(1 for w in response_words if w in context_lower)
        ratio = supported_count / len(response_words)
        
        # Query-aware thresholds
        if query_type in ["simple", "SIMPLE"]:
            required = self.overlap_simple
        elif query_type in ["config", "CONFIGURATION"]:
            required = self.overlap_config
        else:
            required = self.overlap_default
        
        logger.debug(
            f"[VALIDATION] Overlap: {ratio:.2f} ({supported_count}/{len(response_words)}) "
            f"Required: {required}"
        )
        
        if ratio >= required:
            return cleaned, SafetyResult(passed=True)
        
        return self.REFUSAL_RESPONSE, SafetyResult(
            passed=False,
            code=RefusalCode.VALIDATION_FAILED,
            reason=f"Keyword overlap {ratio:.2f} below threshold {required}.",
            details={"ratio": ratio, "required": required, "supported": supported_count}
        )
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def is_refusal(self, text: str) -> bool:
        """Check if text is a refusal response."""
        if not text:
            return False
        for r in self.REFUSAL_STRINGS:
            if r.lower() in text.lower():
                return True
        return False
    
    def clean_response(self, response: str) -> str:
        """Remove thinking patterns and chain-of-thought from response."""
        if not response:
            return response
        
        cleaned = response
        for pattern in self._thinking_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Remove emojis
        cleaned = cleaned.encode('ascii', 'ignore').decode('ascii')
        
        return cleaned.strip()
    
    def should_cache(self, result: str) -> bool:
        """Check if a result should be cached (not a refusal)."""
        return not self.is_refusal(result)
    
    def get_refusal_response(self) -> str:
        """Get the standard refusal response."""
        return self.REFUSAL_RESPONSE


# Global instance
_policy: Optional[SafetyPolicy] = None


def get_safety_policy(config=None) -> SafetyPolicy:
    """Get or create the global safety policy instance."""
    global _policy
    if _policy is None:
        _policy = SafetyPolicy(config)
    return _policy

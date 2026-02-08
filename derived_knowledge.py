"""
Derived Knowledge Allowance Layer
=================================
Implements intelligent handling of implicitly documented Nokia GPON concepts.

This module allows the system to provide derived explanations for concepts
like T-CONT, alloc-id, GEM port that are REFERENCED throughout the Nokia
documentation but not FORMALLY DEFINED.

Key Features:
- Concept Coverage Analyzer: Counts mentions across chunks and pages
- Derived Concept Policy: Strict rules for when derived answers are allowed
- Zero hallucination guarantee: Only uses existing document content
"""

import re
from typing import Dict, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class DerivedAnswerType(Enum):
    """Types of derived answers."""
    DEFINITION = "definition"
    ROLE_EXPLANATION = "role_explanation"
    CONTEXT_SUMMARY = "context_summary"


@dataclass
class ConceptCoverage:
    """
    Result of analyzing a concept's presence in the documentation.
    """
    concept: str
    chunk_count: int = 0
    page_count: int = 0
    unique_pages: Set[int] = field(default_factory=set)
    context_snippets: List[str] = field(default_factory=list)
    is_derived_eligible: bool = False
    eligibility_reason: str = ""


# =============================================================================
# DERIVED CONCEPT POLICY - Strict thresholds
# =============================================================================

DERIVED_CONCEPT_POLICY = {
    # Minimum thresholds for derived answer eligibility
    "min_chunk_count": 5,      # Concept must appear in ≥5 chunks
    "min_page_count": 3,       # Concept must appear on ≥3 unique pages
    "max_snippet_length": 300, # Max chars per context snippet
    "max_snippets": 5,         # Maximum snippets to collect
    
    # Allowed query types for derived answers (DEFINITION ONLY)
    "allowed_query_types": ["simple", "general"],  # NOT config, NOT troubleshooting
    
    # Blocked patterns - NEVER allow derived answers for these
    "blocked_patterns": [
        r'\b(how\s+to\s+configure)',
        r'\b(cli\s+command)',
        r'\b(command\s+line)',
        r'\b(step\s+by\s+step)',
        r'\b(create|delete|modify|set|assign)\s+\w+',
        r'\b(configure|setup|install)',
        r'\b(default|interval|numeric|value|number)\s+',
        r'\bvs\b',  # Cross-vendor comparisons
        r'\bhuawei\b',
        r'\bcisco\b',
        r'\bzte\b',
    ],
    
    # Nokia GPON concepts commonly referenced but not formally defined
    "known_implicit_concepts": {
        't-cont', 'tcont', 'alloc-id', 'allocid', 'gem', 'gemport', 'gem-port',
        'ploam', 'omci', 'dba', 'ranging', 'activation', 'deactivation',
        'upstream', 'downstream', 'onu-id', 'pon-id', 'serial-number',
        'equipment-id', 'version-id', 'gpon', 'xgs-pon', 'ont', 'olt'
    }
}


# =============================================================================
# CONCEPT COVERAGE ANALYZER
# =============================================================================

def analyze_concept_coverage(
    concept: str,
    metadata_store: List[Dict],
    context_window: int = 2
) -> ConceptCoverage:
    """
    Analyze how frequently a concept appears in the documentation.
    
    Args:
        concept: The concept/term to analyze (e.g., "T-CONT")
        metadata_store: List of chunk metadata from FAISS index
        context_window: Number of sentences before/after to extract
    
    Returns:
        ConceptCoverage object with analysis results
    """
    result = ConceptCoverage(concept=concept)
    concept_lower = concept.lower().replace('-', '').replace('_', '')
    
    # Build regex pattern for flexible matching
    # Match: t-cont, tcont, T-CONT, TCONT, t_cont, etc.
    pattern = re.compile(
        r'\b' + re.escape(concept_lower).replace(r'\ ', r'[\s\-_]*') + r'\b',
        re.IGNORECASE
    )
    
    seen_pages = set()
    snippets_collected = 0
    
    for chunk_meta in metadata_store:
        text = chunk_meta.get('enriched_text', chunk_meta.get('original_text', ''))
        text_normalized = text.lower().replace('-', '').replace('_', '')
        
        if pattern.search(text_normalized):
            result.chunk_count += 1
            
            # Extract page number
            page = chunk_meta.get('metadata', {}).get('page', chunk_meta.get('page', 0))
            if isinstance(page, str):
                try:
                    page = int(page)
                except:
                    page = 0
            
            if page > 0:
                seen_pages.add(page)
            
            # Extract context snippet around the concept
            if snippets_collected < DERIVED_CONCEPT_POLICY["max_snippets"]:
                snippet = extract_context_snippet(text, concept, context_window)
                if snippet and len(snippet) <= DERIVED_CONCEPT_POLICY["max_snippet_length"]:
                    result.context_snippets.append(snippet)
                    snippets_collected += 1
    
    result.page_count = len(seen_pages)
    result.unique_pages = seen_pages
    
    # Determine eligibility
    result.is_derived_eligible = (
        result.chunk_count >= DERIVED_CONCEPT_POLICY["min_chunk_count"] and
        result.page_count >= DERIVED_CONCEPT_POLICY["min_page_count"]
    )
    
    if result.is_derived_eligible:
        result.eligibility_reason = (
            f"Found in {result.chunk_count} chunks across {result.page_count} pages "
            f"(thresholds: chunks≥{DERIVED_CONCEPT_POLICY['min_chunk_count']}, "
            f"pages≥{DERIVED_CONCEPT_POLICY['min_page_count']})"
        )
    else:
        result.eligibility_reason = (
            f"Insufficient coverage: {result.chunk_count} chunks, {result.page_count} pages "
            f"(need: chunks≥{DERIVED_CONCEPT_POLICY['min_chunk_count']}, "
            f"pages≥{DERIVED_CONCEPT_POLICY['min_page_count']})"
        )
    
    return result


def extract_context_snippet(text: str, concept: str, window: int = 2) -> Optional[str]:
    """
    Extract sentences surrounding a concept mention.
    
    Args:
        text: Full text to search
        concept: The concept to find
        window: Number of sentences before/after to include
    
    Returns:
        Context snippet or None if not found
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    concept_lower = concept.lower()
    
    for i, sentence in enumerate(sentences):
        if concept_lower in sentence.lower():
            # Get surrounding sentences
            start = max(0, i - window)
            end = min(len(sentences), i + window + 1)
            snippet = ' '.join(sentences[start:end])
            return snippet.strip()
    
    return None


# =============================================================================
# QUERY CLASSIFICATION FOR DERIVED ANSWERS
# =============================================================================

def is_query_eligible_for_derived(query: str, query_type: str) -> tuple:
    """
    Check if a query is eligible for derived answer based on policy.
    
    Args:
        query: The user's query
        query_type: The classified query type (simple, config, etc.)
    
    Returns:
        Tuple of (is_eligible: bool, reason: str)
    """
    query_lower = query.lower()
    
    # Check query type
    if query_type.lower() not in DERIVED_CONCEPT_POLICY["allowed_query_types"]:
        return (False, f"Query type '{query_type}' not allowed for derived answers")
    
    # Check blocked patterns
    for pattern in DERIVED_CONCEPT_POLICY["blocked_patterns"]:
        if re.search(pattern, query_lower):
            return (False, f"Query matches blocked pattern: {pattern}")
    
    # Check if asking for CLI/commands
    cli_indicators = ['cli', 'command', 'syntax', 'configure', 'create', 'delete', 'modify']
    for indicator in cli_indicators:
        if indicator in query_lower:
            return (False, f"Query contains CLI/configuration indicator: '{indicator}'")
    
    return (True, "Query eligible for derived answer")


def extract_concept_from_query(query: str) -> Optional[str]:
    """
    Extract the main concept being asked about from a query.
    
    Args:
        query: The user's query (e.g., "What is T-CONT?")
    
    Returns:
        The extracted concept or None
    """
    # Common question patterns
    patterns = [
        r"what\s+is\s+(?:a\s+|an\s+)?([a-z0-9\-_]+)",
        r"what\s+does\s+([a-z0-9\-_]+)\s+mean",
        r"define\s+([a-z0-9\-_]+)",
        r"explain\s+([a-z0-9\-_]+)",
        r"what\s+are\s+([a-z0-9\-_]+)",
    ]
    
    query_lower = query.lower()
    
    for pattern in patterns:
        match = re.search(pattern, query_lower)
        if match:
            return match.group(1).upper()  # Return in uppercase for consistency
    
    return None


# =============================================================================
# DERIVED ANSWER GENERATOR
# =============================================================================

def generate_derived_answer_context(coverage: ConceptCoverage) -> str:
    """
    Generate the context/prompt addition for a derived answer.
    
    This creates structured content from the coverage analysis that
    can be used by the LLM to generate a derived explanation.
    
    Args:
        coverage: The ConceptCoverage result
    
    Returns:
        Formatted context string for LLM
    """
    if not coverage.is_derived_eligible:
        return ""
    
    lines = [
        f"=== DERIVED KNOWLEDGE CONTEXT FOR: {coverage.concept} ===",
        f"This concept appears in {coverage.chunk_count} documentation chunks",
        f"across {coverage.page_count} unique pages.",
        "",
        "Relevant context snippets from documentation:",
        ""
    ]
    
    for i, snippet in enumerate(coverage.context_snippets, 1):
        lines.append(f"[{i}] {snippet}")
        lines.append("")
    
    lines.extend([
        "INSTRUCTIONS FOR DERIVED ANSWER:",
        "1. Explain the ROLE and FUNCTION of this concept based on context above",
        "2. DO NOT invent commands or CLI syntax",
        "3. DO NOT provide configuration steps",
        "4. DO NOT use numbers/values not present in snippets",
        "5. Begin response with: 'Derived from Nokia GPON documentation context:'",
        ""
    ])
    
    return "\n".join(lines)


# =============================================================================
# DERIVED ANSWER RESPONSE FORMATTER
# =============================================================================

DERIVED_ANSWER_PREFIX = "**Derived from Nokia GPON documentation context:**\n\n"

DERIVED_ANSWER_SUFFIX = (
    "\n\n---\n"
    "*This concept is referenced throughout the documentation but not formally defined. "
    "Providing a derived explanation strictly based on available context.*"
)

DERIVED_REFUSAL_MESSAGE = (
    "This concept is referenced in the documentation but not formally defined. "
    "However, there is insufficient context to provide a reliable derived explanation. "
    "Please consult the official Nokia documentation for a formal definition."
)


def format_derived_response(response: str, coverage: ConceptCoverage) -> str:
    """
    Format a derived answer with proper labeling.
    
    Args:
        response: The raw LLM response
        coverage: The coverage analysis
    
    Returns:
        Formatted response with derived answer labels
    """
    # Ensure the response doesn't already have the prefix
    if "Derived from Nokia" in response:
        # Already formatted, just add suffix
        return response + DERIVED_ANSWER_SUFFIX
    
    return DERIVED_ANSWER_PREFIX + response + DERIVED_ANSWER_SUFFIX


# =============================================================================
# MAIN INTERFACE - Should Attempt Derived Answer
# =============================================================================

def should_attempt_derived_answer(
    query: str,
    query_type: str,
    context: str,
    metadata_store: List[Dict]
) -> tuple:
    """
    Determine if we should attempt a derived answer for a query.
    
    This is the main entry point for the derived knowledge layer.
    
    Args:
        query: User's query
        query_type: Classified query type
        context: Retrieved context (may indicate refusal)
        metadata_store: FAISS metadata for coverage analysis
    
    Returns:
        Tuple of (should_attempt: bool, coverage: ConceptCoverage or None, reason: str)
    """
    # Check if the retrieval already refused
    if "INSUFFICIENT DOCUMENTATION CONTEXT" not in context:
        # Retrieval succeeded, no need for derived answer
        return (False, None, "Retrieval succeeded - no derived answer needed")
    
    # Check if query is eligible
    is_eligible, eligibility_reason = is_query_eligible_for_derived(query, query_type)
    if not is_eligible:
        return (False, None, eligibility_reason)
    
    # Extract the concept being asked about
    concept = extract_concept_from_query(query)
    if not concept:
        return (False, None, "Could not extract concept from query")
    
    # Check if this is a known implicit concept
    concept_lower = concept.lower()
    is_known_implicit = any(
        kc in concept_lower or concept_lower in kc 
        for kc in DERIVED_CONCEPT_POLICY["known_implicit_concepts"]
    )
    
    if not is_known_implicit:
        return (False, None, f"Concept '{concept}' not in known implicit concepts list")
    
    # Analyze concept coverage
    coverage = analyze_concept_coverage(concept, metadata_store)
    
    if not coverage.is_derived_eligible:
        return (False, coverage, coverage.eligibility_reason)
    
    return (True, coverage, f"Derived answer eligible: {coverage.eligibility_reason}")


# =============================================================================
# SYSTEM PROMPT FOR DERIVED ANSWERS
# =============================================================================

DERIVED_ANSWER_SYSTEM_PROMPT = """You are a Nokia Technical Assistant providing a DERIVED explanation.

IMPORTANT: The user is asking about a concept ({concept}) that is REFERENCED throughout 
the Nokia GPON documentation but is NOT formally defined. You must provide an explanation 
based ONLY on the context provided below.

{derived_context}

STRICT RULES:
1. Begin your response with: "Derived from Nokia GPON documentation context:"
2. Explain what the concept IS and what ROLE it plays based on the snippets
3. DO NOT invent CLI commands or syntax
4. DO NOT provide step-by-step configuration instructions
5. DO NOT use specific numbers or values not present in the context
6. DO NOT speculate or add information beyond what's in the context
7. If the context is insufficient, say so honestly

The user asked: {query}
"""

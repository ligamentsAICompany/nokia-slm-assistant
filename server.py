"""
Nokia SLM Flask Server
======================
Production-ready Flask API server for the Nokia SLM system.
"""

from flask import Flask, request, jsonify, render_template
import os
import sys
import time
import logging

# Ensure backend can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import configuration first
try:
    from config import load_config
    from observability import setup_logging, Metrics, HealthChecker, QueryTracer
    CONFIG = load_config()
    setup_logging(level=CONFIG.log_level, format_type=CONFIG.log_format)
except ImportError:
    CONFIG = None
    logging.basicConfig(level=logging.INFO)

from slm_logic import SLMBackend, QueryType

# Import derived knowledge layer
try:
    from derived_knowledge import (
        should_attempt_derived_answer,
        generate_derived_answer_context,
        format_derived_response,
        validate_derived_response,
        DERIVED_ANSWER_SYSTEM_PROMPT,
        DERIVED_REFUSAL_MESSAGE,
        DERIVED_UI_BANNER
    )
    DERIVED_KNOWLEDGE_AVAILABLE = True
except ImportError:
    DERIVED_KNOWLEDGE_AVAILABLE = False

logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")

# Initialize Backend
logger.info("Initializing SLM Backend...")
backend = SLMBackend()
backend.load_resources()

# Step 5: Warm-up LLM at startup (eliminates first-query delay)
logger.info("[STARTUP] Warming up LLM...")
try:
    warmup_response = backend.generate_response(
        messages=[{"role": "user", "content": "Hello"}],
        query_type=None
    )
    if "SYSTEM_ERROR" not in warmup_response:
        logger.info("[STARTUP] LLM warm-up successful")
    else:
        logger.warning("[STARTUP] LLM warm-up failed (will retry on first query)")
except Exception as e:
    logger.warning(f"[STARTUP] LLM warm-up skipped: {e}")

logger.info("Backend Ready.")

# --- DYNAMIC SYSTEM PROMPTS ---
# --- DYNAMIC SYSTEM PROMPTS ---
# Updated to support derived answers with proper labeling

SYSTEM_PROMPTS = {
    QueryType.SIMPLE: """You are a Nokia Technical Assistant answering questions from documentation.

The user asked: {query}

INSTRUCTIONS:
1. Read the REFERENCE DATA below carefully.
2. Answer the question using information from the reference data.
3. Be concise (1-3 sentences for definitions).
4. If related information exists, provide it even if not a perfect match.
5. ONLY say "INSUFFICIENT DOCUMENTATION CONTEXT" if there is truly no relevant information.

REFERENCE DATA:
{context}
""",

    QueryType.CONFIGURATION: """You are a Senior Nokia TAC Engineer. You ONLY provide CLI commands that are EXPLICITLY documented.

The user asked: {query}

EXPLICIT CLI ONLY POLICY:
You must ONLY return CLI commands if they appear EXACTLY in the REFERENCE DATA below.

RULE 1 - EXPLICIT CLI FOUND:
If exact CLI commands/syntax appear in the reference data:
**📋 Documented CLI Commands:**
```
[copy exact commands from documentation - do NOT modify syntax]
```

Then explain any documented parameters or prerequisites.

RULE 2 - NO EXPLICIT CLI FOUND → REFUSE:
If the reference data does NOT contain exact CLI syntax for this task:
- Do NOT invent or derive CLI commands
- Do NOT provide generic command patterns
- Do NOT guess syntax based on similar commands
- RESPOND WITH: "INSUFFICIENT DOCUMENTATION CONTEXT: The requested configuration commands are not explicitly documented in the available reference material. Please consult the official Nokia CLI reference guide."

CRITICAL RESTRICTIONS:
- NEVER invent CLI syntax
- NEVER provide step numbers without documented commands
- NEVER use phrases like "typically", "usually", "would be"
- NEVER provide commands from external knowledge

REFERENCE DATA:
{context}
""",

    QueryType.TROUBLESHOOTING: """You are a Nokia Troubleshooting Expert.

The user asked: {query}

INSTRUCTIONS:
1. Find relevant troubleshooting information in the reference data.
2. Provide:
   - Possible Causes (from documentation)
   - Diagnostic Steps (from documentation)
   - Resolution/Fix (from documentation)
3. If partial information is available, provide what you can and label as:
   **⚠️ Partial information - see documentation for complete steps**
4. Only refuse if no relevant troubleshooting info exists.

REFERENCE DATA:
{context}
""",

    QueryType.LOG_ANALYSIS: """You are a Nokia Log Analysis Expert.

The user provided log data and asked: {query}

INSTRUCTIONS:
1. Analyze the log data using the reference documentation.
2. Identify:
   - Alarm/Error type
   - Probable root cause
   - Affected components
3. Provide:
   - Meaning of the error/alarm
   - Documented resolution steps
4. Use structured format for clarity.

REFERENCE DATA:
{context}
""",

    QueryType.GENERAL: """You are a Nokia Technical Assistant.

The user asked: {query}

INSTRUCTIONS:
1. Answer using the reference data provided.
2. Be helpful and provide relevant information.
3. If the exact answer isn't there but related info exists, share it with the label:
   **ℹ️ Related information from documentation:**
4. Only say "INSUFFICIENT DOCUMENTATION CONTEXT" if nothing relevant is found.

REFERENCE DATA:
{context}
"""
}

# --- NOKIA ENTITY KEYWORDS FOR SAFETY CHECK ---
NOKIA_ENTITIES = {
    'ont', 'olt', 'gpon', 'xgs-pon', 'pon', 't-cont', 'tcont', 'gem', 'gemport',
    'alloc-id', 'vlan', 'service-port', 'dba', 'bandwidth', 'profile', 'qos',
    'upstream', 'downstream', 'onu', 'sfp', 'fiber', 'optical', 'ethernet',
    'g-984', 'g-987', 'g-988', 'omci', 'ploam', 'ranging', 'activation',
    'deactivation', 'los', 'lof', 'lom', 'dying-gasp', 'rogue', 'password',
    'registration', 'serial-number', 'equipment-id', 'version', 'firmware',
    'management', 'snmp', 'tr-069', 'cwmp', 'netconf', 'cli', 'tl1',
    'isam', 'altiplano', '7360', '7362', '7368', 'g-240', 'g-140'
}

NOKIA_COMMAND_KEYWORDS = {
    'configure', 'show', 'display', 'create', 'delete', 'modify', 'set',
    'enable', 'disable', 'add', 'remove', 'interface', 'port', 'service',
    'bridge', 'router', 'vlan', 'qos', 'bandwidth', 'profile', 'equipment'
}


def validate_nokia_context(response_text: str) -> bool:
    """
    Hard safety check for responses.
    Returns True if response contains at least 1 Nokia entity.
    """
    response_lower = response_text.lower()
    return any(entity in response_lower for entity in NOKIA_ENTITIES)


def classify_response_type(response_text: str) -> str:
    """
    Classify response as 'explicit', 'derived', or 'refusal' for UI display.
    """
    if "INSUFFICIENT DOCUMENTATION CONTEXT" in response_text:
        return "refusal"
    elif "Derived from Documentation" in response_text or "⚠️" in response_text:
        return "derived"
    elif "📋 Documented CLI" in response_text or "```" in response_text:
        return "explicit"
    else:
        return "derived"  # Default to derived if unclear


# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def home():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Get system status."""
    return jsonify({
        "vectors": backend.vector_index.ntotal if backend.vector_index else 0,
        "chunks": len(backend.metadata_store),
        "model": backend.active_model[:30] + "..." if len(backend.active_model) > 30 else backend.active_model,
        "bm25_enabled": bool(backend.bm25),
        "cross_encoder_enabled": bool(backend.cross_encoder),
        "graph_enabled": bool(backend.graph)
    })


@app.route('/api/health')
def health():
    """Health check endpoint."""
    try:
        if CONFIG:
            health_status = HealthChecker.run_all_checks(CONFIG)
            return jsonify(health_status.to_dict()), 200 if health_status.status == "healthy" else 503
        else:
            # Basic health check without config
            return jsonify({
                "status": "healthy" if backend.vector_index else "unhealthy",
                "checks": {
                    "faiss_index": backend.vector_index is not None,
                    "metadata": len(backend.metadata_store) > 0
                },
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503


@app.route('/api/metrics')
def metrics():
    """Metrics endpoint."""
    try:
        return jsonify(Metrics.get_metrics())
    except Exception as e:
        logger.error(f"Metrics failed: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/stats')
def stats():
    """Get comprehensive system statistics."""
    return jsonify(backend.get_all_stats())


@app.route('/api/chat', methods=['POST'])
def chat():
    """Main chat endpoint for query processing."""
    data = request.json
    user_query = data.get('query')
    
    if not user_query:
        return jsonify({"error": "No query provided"}), 400
    
    start_time = time.time()
    
    # Use QueryTracer if available
    try:
        tracer = QueryTracer(user_query)
        trace = tracer.trace(user_query).__enter__()
    except:
        trace = None
    
    try:
        # 1. Retrieve Context with Query Classification
        context, query_type = backend.retrieve_context(user_query)
        
        if trace:
            trace.query_type = query_type.value
        
        # DERIVED KNOWLEDGE LAYER: Intercept refusals for implicit concepts
        if "INSUFFICIENT DOCUMENTATION CONTEXT" in context:
            if DERIVED_KNOWLEDGE_AVAILABLE:
                should_use_dcal, coverage, reason = should_attempt_derived_answer(
                    query=user_query,
                    query_type=query_type.value,
                    context=context,
                    metadata_store=backend.metadata_store
                )
                
                if should_use_dcal:
                    logger.info(f"[DCAL] Attempting derived answer: {reason}")
                    derived_context = generate_derived_answer_context(coverage)
                    
                    messages = [
                        {
                            "role": "system", 
                            "content": DERIVED_ANSWER_SYSTEM_PROMPT.format(
                                concept=coverage.concept,
                                derived_context=derived_context,
                                query=user_query
                            )
                        },
                        {"role": "user", "content": user_query}
                    ]
                    
                    # Generate derived response
                    raw_response = backend.generate_response(messages, query_type)
                    
                    # POST-GENERATION VALIDATION (Speculation words, CLI patterns)
                    is_valid, violations = validate_derived_response(raw_response)
                    if not is_valid:
                        logger.warning(f"[DCAL] Validation failed: {violations}")
                        return _refuse_chat(context, query_type, start_time, trace)
                    
                    # Ensure it contains at least some Nokia context
                    if not validate_nokia_context(raw_response):
                        logger.warning(f"[DCAL] Nokia context check failed")
                        return _refuse_chat(context, query_type, start_time, trace)

                    formatted_response = format_derived_response(raw_response, coverage)
                    
                    if trace:
                        trace.response_grounded = True
                        trace.complete()
                    
                    return jsonify({
                        "response": formatted_response,
                        "query_type": query_type.value,
                        "response_type": "derived",
                        "status": "derived_explanation",
                        "ui_banner": DERIVED_UI_BANNER,
                        "derived_info": {
                            "concept": coverage.concept,
                            "chunk_count": coverage.chunk_count,
                            "page_count": coverage.page_count
                        },
                        "latency_ms": round((time.time() - start_time) * 1000, 1)
                    })

            # Standard refusal if DCAL not used/available
            return _refuse_chat(context, query_type, start_time, trace)

        # 2. Select Dynamic System Prompt based on Query Type
        system_prompt = SYSTEM_PROMPTS.get(query_type, SYSTEM_PROMPTS[QueryType.GENERAL])
        system_prompt = system_prompt.format(context=context, query=user_query)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        # 3. Generate Response
        response_text = backend.generate_response(messages, query_type)
        
        if trace:
            trace.llm_used = backend.active_model
        
        # Check for system error
        if response_text.startswith("SYSTEM_ERROR:"):
            if trace:
                trace.refusal = True
                trace.refusal_code = "SYSTEM_ERROR"
                trace.complete()
            
            return jsonify({
                "response": "The system is experiencing high load. Please try again in a moment.",
                "query_type": query_type.value,
                "grounded": False,
                "confidence": "system_retry",
                "status": "system_retry",
                "context_preview": "LLM temporarily unavailable",
                "latency_ms": round((time.time() - start_time) * 1000, 1)
            })
        
        # 4. Post-Generation Validation
        validated_response = backend.validate_response(response_text, context, query_type)
        
        is_grounded = "INSUFFICIENT DOCUMENTATION CONTEXT" not in validated_response
        
        # 5. HARD SAFETY CHECK for configuration queries
        # Ensure derived responses contain Nokia-specific content
        if is_grounded and query_type == QueryType.CONFIGURATION:
            if not validate_derived_response(validated_response):
                # Force refusal if response lacks Nokia entities/commands
                validated_response = "INSUFFICIENT DOCUMENTATION CONTEXT: Unable to provide configuration guidance. The retrieved documentation does not contain sufficient Nokia-specific configuration information for this query."
                is_grounded = False
                if trace:
                    trace.refusal = True
                    trace.refusal_code = "DERIVED_SAFETY_FAILED"
        
        # Classify response type for UI
        response_type = classify_response_type(validated_response)
        
        if trace:
            trace.response_grounded = is_grounded
            if not is_grounded:
                trace.refusal = True
                trace.refusal_code = trace.refusal_code if hasattr(trace, 'refusal_code') and trace.refusal_code else "VALIDATION_FAILED"
            trace.complete()
        
        latency_ms = round((time.time() - start_time) * 1000, 1)
        
        # UI text distinction based on response type
        if response_type == "derived":
            confidence_label = "derived"
            status_label = "derived_guidance"
        elif response_type == "explicit":
            confidence_label = "high"
            status_label = "success"
        else:
            confidence_label = "refusal"
            status_label = "no_doc"
        
        return jsonify({
            "response": validated_response,
            "query_type": query_type.value,
            "response_type": response_type,  # NEW: "explicit", "derived", or "refusal"
            "grounded": is_grounded,
            "confidence": confidence_label,
            "status": status_label,
            "context_preview": context[:500] + "...",
            "latency_ms": latency_ms
        })
        
    except Exception as e:
        logger.exception(f"Chat error: {e}")
        if trace:
            trace.refusal = True
            trace.refusal_code = "EXCEPTION"
            trace.complete()
        
        return jsonify({
            "error": str(e),
            "status": "error",
            "latency_ms": round((time.time() - start_time) * 1000, 1)
        }), 500


def _refuse_chat(context, query_type, start_time, trace):
    """Helper for refusal responses."""
    if trace:
        trace.refusal = True
        trace.refusal_code = "RETRIEVAL_REFUSED"
        trace.complete()
    
    return jsonify({
        "response": context,
        "query_type": query_type.value,
        "response_type": "refusal",
        "grounded": False,
        "confidence": "refusal",
        "context_preview": "Refused due to insufficient context.",
        "latency_ms": round(float(time.time() - start_time) * 1000, 1)
    })


@app.route('/api/cache/clear', methods=['POST'])
def clear_cache():
    """Clear the query cache."""
    success = backend.clear_cache()
    return jsonify({"success": success})


# =============================================================================
# STARTUP
# =============================================================================

if __name__ == '__main__':
    host = CONFIG.server_host if CONFIG else "0.0.0.0"
    port = CONFIG.server_port if CONFIG else 5000
    debug = CONFIG.server_debug if CONFIG else False
    
    logger.info(f"Starting server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)

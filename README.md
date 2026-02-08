# Nokia SLM Assistant

> **Enterprise-grade Small Language Model for Nokia GPON/ONT Technical Documentation**

A production-ready RAG (Retrieval-Augmented Generation) system that answers technical questions from Nokia GPON/ONT documentation with strict safety guardrails and hallucination prevention.

---

## Features

| Feature | Description |
|---------|-------------|
| **Hybrid Search** | Combines semantic vector search (FastEmbed) + BM25 keyword search + RRF fusion |
| **Cross-Encoder Reranking** | Uses cross-encoder for relevance scoring with rank-based gating |
| **Safety Policy** | Centralized pre/post validation with injection detection and hallucination prevention |
| **Query Classification** | Automatic routing (SIMPLE, CONFIG, TROUBLE, LOG) with query-aware handling |
| **Alarm Intelligence** | Specialized parsing and structured responses for alarm/log queries |
| **Observability** | Structured JSON logging, metrics collection, health endpoints |
| **LLM Fallback** | Primary (LM Studio) + Fallback (Ollama) with adaptive timeouts and retry |

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env with your LM Studio URL and paths
```

### 3. Ingest Documents

Place your Nokia PDF documentation in the `documents/` folder, then run:

```bash
python -m ingestion.run_ingestion
```

### 4. Start Server

```bash
python server.py
```

Open http://localhost:5000 in your browser.

---

## Project Structure

```
nokia-slm-assistant/
├── server.py              # Flask API + HTML serving
├── slm_logic.py           # Core RAG engine
├── safety_policy.py       # Centralized safety rules
├── observability.py       # Logging, metrics, health checks
├── alarm_intelligence.py  # Alarm/log parsing
├── optimizations.py       # Caching, context window management
├── config/
│   ├── __init__.py
│   └── settings.py        # Environment-based configuration
├── ingestion/
│   ├── __init__.py
│   ├── pdf_loader.py      # PyMuPDF loader
│   ├── chunker.py         # Deterministic chunking
│   ├── embedder.py        # FastEmbed wrapper
│   ├── indexer.py         # FAISS index builder
│   ├── graph_builder.py   # Knowledge graph construction
│   └── run_ingestion.py   # CLI entry point
├── templates/
│   └── index.html         # Web UI
├── static/
│   ├── styles.css         # Styling
│   └── app.js             # Frontend logic
├── tests/
│   ├── test_safety.py     # Safety policy tests
│   └── ...
├── .env.example           # Environment template
└── requirements.txt       # Dependencies
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serve web UI |
| `/api/chat` | POST | Process query and return answer |
| `/api/status` | GET | System status (model, vectors, etc.) |
| `/api/health` | GET | Health check for monitoring |
| `/api/metrics` | GET | Prometheus-style metrics |
| `/api/stats` | GET | Comprehensive system statistics |

---

## Configuration

All settings are configurable via environment variables. See `.env.example` for the full list.

Key settings:

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://localhost:8103/v1` | Primary LLM endpoint |
| `OLLAMA_URL` | `http://localhost:11434/api/chat` | Fallback LLM endpoint |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | Embedding model (384-dim) |
| `CHUNK_SIZE` | `512` | Characters per chunk |
| `MIN_CONFIDENCE_THRESHOLD` | `0.05` | Minimum retrieval confidence |

---

## Safety Features

1. **Pre-Retrieval Checks**: Query length, injection detection, out-of-scope filtering
2. **Post-Retrieval Checks**: Ensures retrieval hits before LLM call
3. **Post-Generation Validation**: Keyword overlap validation to detect hallucination
4. **Strict Refusal**: Returns "INSUFFICIENT DOCUMENTATION CONTEXT" when uncertain

---

## License

Proprietary - Nokia / Ligaments AI Company

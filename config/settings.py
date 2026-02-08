"""
Nokia SLM Configuration
=======================
Centralized, environment-based configuration management.
All paths, URLs, and parameters are configurable via environment variables.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _get_env(key: str, default: str) -> str:
    """Get environment variable with default."""
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    """Get integer environment variable with default."""
    return int(os.environ.get(key, str(default)))


def _get_env_float(key: str, default: float) -> float:
    """Get float environment variable with default."""
    return float(os.environ.get(key, str(default)))


def _get_env_bool(key: str, default: bool) -> bool:
    """Get boolean environment variable with default."""
    val = os.environ.get(key, str(default)).lower()
    return val in ("true", "1", "yes", "on")


@dataclass
class SLMConfig:
    """
    Nokia SLM Configuration.
    All settings can be overridden via environment variables.
    """
    
    # === Paths ===
    base_dir: str = field(default_factory=lambda: _get_env(
        "SLM_BASE_DIR", 
        str(Path(__file__).parent.parent.absolute())
    ))
    data_dir: str = field(default_factory=lambda: _get_env(
        "SLM_DATA_DIR",
        str(Path(_get_env("SLM_BASE_DIR", str(Path(__file__).parent.parent.absolute()))) / "data")
    ))
    cache_dir: str = field(default_factory=lambda: _get_env(
        "SLM_CACHE_DIR",
        str(Path(_get_env("SLM_BASE_DIR", str(Path(__file__).parent.parent.absolute()))) / "cache")
    ))
    documents_dir: str = field(default_factory=lambda: _get_env(
        "SLM_DOCUMENTS_DIR",
        str(Path(_get_env("SLM_BASE_DIR", str(Path(__file__).parent.parent.absolute()))) / "documents")
    ))
    
    # === Index Paths ===
    faiss_index_path: str = field(default_factory=lambda: _get_env(
        "SLM_FAISS_INDEX",
        str(Path(_get_env("SLM_DATA_DIR", "data")) / "nokia_vector_index.faiss")
    ))
    metadata_path: str = field(default_factory=lambda: _get_env(
        "SLM_METADATA_PATH",
        str(Path(_get_env("SLM_DATA_DIR", "data")) / "nokia_vector_meta.pkl")
    ))
    graph_path: str = field(default_factory=lambda: _get_env(
        "SLM_GRAPH_PATH",
        str(Path(_get_env("SLM_DATA_DIR", "data")) / "nokia_graph.gml")
    ))
    query_cache_path: str = field(default_factory=lambda: _get_env(
        "SLM_QUERY_CACHE",
        str(Path(_get_env("SLM_CACHE_DIR", "cache")) / "query_cache.json")
    ))
    
    # === LLM Configuration ===
    lm_studio_url: str = field(default_factory=lambda: _get_env(
        "LM_STUDIO_URL",
        "http://localhost:8103/v1"
    ))
    ollama_url: str = field(default_factory=lambda: _get_env(
        "OLLAMA_URL",
        "http://localhost:11434/api/chat"
    ))
    ollama_model: str = field(default_factory=lambda: _get_env(
        "OLLAMA_MODEL",
        "llama3.2"
    ))
    
    # === LLM Parameters ===
    llm_temperature: float = field(default_factory=lambda: _get_env_float(
        "LLM_TEMPERATURE", 0.2
    ))
    llm_max_tokens: int = field(default_factory=lambda: _get_env_int(
        "LLM_MAX_TOKENS", 1000
    ))
    llm_timeout_simple: int = field(default_factory=lambda: _get_env_int(
        "LLM_TIMEOUT_SIMPLE", 60
    ))
    llm_timeout_complex: int = field(default_factory=lambda: _get_env_int(
        "LLM_TIMEOUT_COMPLEX", 120
    ))
    llm_max_retries: int = field(default_factory=lambda: _get_env_int(
        "LLM_MAX_RETRIES", 2
    ))
    
    # === Embedding Configuration ===
    embedding_model: str = field(default_factory=lambda: _get_env(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    ))
    embedding_dim: int = field(default_factory=lambda: _get_env_int(
        "EMBEDDING_DIM", 384
    ))
    
    # === Chunking Configuration ===
    chunk_size: int = field(default_factory=lambda: _get_env_int(
        "CHUNK_SIZE", 512
    ))
    chunk_overlap: int = field(default_factory=lambda: _get_env_int(
        "CHUNK_OVERLAP", 64
    ))
    
    # === Retrieval Configuration ===
    vector_top_k: int = field(default_factory=lambda: _get_env_int(
        "VECTOR_TOP_K", 50
    ))
    bm25_top_k: int = field(default_factory=lambda: _get_env_int(
        "BM25_TOP_K", 50
    ))
    rerank_top_k: int = field(default_factory=lambda: _get_env_int(
        "RERANK_TOP_K", 15
    ))
    max_context_chunks: int = field(default_factory=lambda: _get_env_int(
        "MAX_CONTEXT_CHUNKS", 200
    ))
    
    # === Safety Configuration ===
    min_confidence_threshold: float = field(default_factory=lambda: _get_env_float(
        "MIN_CONFIDENCE_THRESHOLD", 0.05
    ))
    validation_overlap_simple: float = field(default_factory=lambda: _get_env_float(
        "VALIDATION_OVERLAP_SIMPLE", 0.08
    ))
    validation_overlap_config: float = field(default_factory=lambda: _get_env_float(
        "VALIDATION_OVERLAP_CONFIG", 0.10
    ))
    validation_overlap_default: float = field(default_factory=lambda: _get_env_float(
        "VALIDATION_OVERLAP_DEFAULT", 0.15
    ))
    
    # === Cache Configuration ===
    cache_ttl_seconds: int = field(default_factory=lambda: _get_env_int(
        "CACHE_TTL_SECONDS", 3600
    ))
    cache_enabled: bool = field(default_factory=lambda: _get_env_bool(
        "CACHE_ENABLED", True
    ))
    
    # === Server Configuration ===
    server_host: str = field(default_factory=lambda: _get_env(
        "SERVER_HOST", "0.0.0.0"
    ))
    server_port: int = field(default_factory=lambda: _get_env_int(
        "SERVER_PORT", 5000
    ))
    server_debug: bool = field(default_factory=lambda: _get_env_bool(
        "SERVER_DEBUG", False
    ))
    
    # === Logging Configuration ===
    log_level: str = field(default_factory=lambda: _get_env(
        "LOG_LEVEL", "INFO"
    ))
    log_format: str = field(default_factory=lambda: _get_env(
        "LOG_FORMAT", "json"  # "json" or "text"
    ))
    log_file: Optional[str] = field(default_factory=lambda: _get_env(
        "LOG_FILE", ""
    ) or None)
    
    # === Cross-Encoder Configuration ===
    cross_encoder_model: str = field(default_factory=lambda: _get_env(
        "CROSS_ENCODER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    ))
    cross_encoder_enabled: bool = field(default_factory=lambda: _get_env_bool(
        "CROSS_ENCODER_ENABLED", True
    ))
    
    # === Context Window ===
    model_context_limits: Dict[str, int] = field(default_factory=lambda: {
        "default": 4096,
        "qwen": 32768,
        "llama-3": 8192,
        "mistral": 32768,
        "phi-3": 128000,
        "gemma": 8192,
        "nemotron": 8192,
    })
    reserved_tokens: int = field(default_factory=lambda: _get_env_int(
        "RESERVED_TOKENS", 1500
    ))
    
    def __post_init__(self):
        """Ensure directories exist."""
        for dir_path in [self.data_dir, self.cache_dir, self.documents_dir]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def get_faiss_index_path(self) -> Path:
        """Get resolved FAISS index path."""
        if os.path.isabs(self.faiss_index_path):
            return Path(self.faiss_index_path)
        return Path(self.data_dir) / "nokia_vector_index.faiss"
    
    def get_metadata_path(self) -> Path:
        """Get resolved metadata path."""
        if os.path.isabs(self.metadata_path):
            return Path(self.metadata_path)
        return Path(self.data_dir) / "nokia_vector_meta.pkl"
    
    def get_graph_path(self) -> Path:
        """Get resolved graph path."""
        if os.path.isabs(self.graph_path):
            return Path(self.graph_path)
        return Path(self.data_dir) / "nokia_graph.gml"


# Global config instance (lazy-loaded)
_config: Optional[SLMConfig] = None


def load_config(force_reload: bool = False) -> SLMConfig:
    """
    Load and return the global configuration.
    
    Args:
        force_reload: If True, reload config from environment.
        
    Returns:
        SLMConfig instance.
    """
    global _config
    if _config is None or force_reload:
        _config = SLMConfig()
    return _config


def get_config() -> SLMConfig:
    """Get the current configuration (alias for load_config)."""
    return load_config()


# Environment setup for TensorFlow/Keras compatibility
def setup_environment():
    """Set up environment variables for ML libraries."""
    os.environ['TF_USE_LEGACY_KERAS'] = '1'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    
    config = load_config()
    os.environ['HF_HOME'] = str(Path(config.cache_dir) / "huggingface")
    os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(Path(config.cache_dir) / "sentence_transformers")

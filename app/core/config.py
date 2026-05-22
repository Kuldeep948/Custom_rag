"""
Application configuration using pydantic-settings.

All settings are loaded from environment variables (.env file).
A structured YAML config file (app/config.yaml) holds the defaults
for Database, Vector DB, and LLM — env vars always take precedence.
"""
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Load YAML config defaults ─────────────────────────────────────────────────
_CONFIG_YAML_PATH = Path(__file__).parent / "config.yaml"


def _load_yaml_defaults() -> dict:
    """Load config.yaml and return a flat dict of defaults."""
    if not _CONFIG_YAML_PATH.exists():
        return {}
    with open(_CONFIG_YAML_PATH, "r") as f:
        data = yaml.safe_load(f) or {}

    flat: dict = {}

    # database section
    db = data.get("database", {})
    flat["database_url"] = db.get("url", flat.get("database_url"))
    flat["database_url_sync"] = db.get("url_sync", flat.get("database_url_sync"))
    flat["db_pool_size"] = db.get("pool_size", 10)
    flat["db_max_overflow"] = db.get("max_overflow", 20)
    flat["db_pool_timeout"] = db.get("pool_timeout", 30)

    # vector_store section
    vs = data.get("vector_store", {})
    flat["qdrant_url"] = vs.get("url", "http://localhost:6333")
    flat["qdrant_collection_name"] = vs.get("collection_name", "rag_knowledge_base")
    flat["qdrant_distance_metric"] = vs.get("distance_metric", "cosine")
    flat["qdrant_on_disk_vectors"] = vs.get("on_disk_vectors", False)

    # embedding section
    emb = data.get("embedding", {})
    flat["embedding_provider"] = emb.get("provider", "sentence_transformer")
    flat["embedding_model"] = emb.get("model", "all-MiniLM-L6-v2")
    flat["embedding_dimension"] = emb.get("dimension", 384)
    flat["st_batch_size"] = emb.get("batch_size", 64)
    flat["st_device"] = emb.get("device", "cpu")
    flat["st_normalize"] = emb.get("normalize_embeddings", True)

    # llm section
    llm = data.get("llm", {})
    flat["llm_provider"] = llm.get("provider", "gemini")
    flat["llm_model"] = llm.get("model", "gemini-1.5-flash")
    flat["llm_max_tokens"] = llm.get("max_tokens", 1024)
    flat["llm_temperature"] = llm.get("temperature", 0.1)

    # chunking section
    chunk = data.get("chunking", {})
    flat["chunk_size"] = chunk.get("size", 512)
    flat["chunk_overlap"] = chunk.get("overlap", 50)
    flat["min_chunk_size"] = chunk.get("min_size", 100)

    # query section
    query = data.get("query", {})
    flat["default_top_k"] = query.get("default_top_k", 5)
    flat["max_top_k"] = query.get("max_top_k", 20)

    return {k: v for k, v in flat.items() if v is not None}


_YAML_DEFAULTS = _load_yaml_defaults()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────
    app_name: str = "RAG Knowledge Base API"
    app_version: str = "1.0.0"
    environment: str = "development"
    debug: bool = False
    secret_key: str = "change-this-secret-key"

    # ── API Authentication ────────────────────────────────────────
    api_key: str = "dev-api-key-change-in-production"

    # ── Database (PostgreSQL) ─────────────────────────────────────
    database_url: str = _YAML_DEFAULTS.get(
        "database_url",
        "postgresql+asyncpg://raguser:ragpassword@localhost:5432/ragdb",
    )
    database_url_sync: str = _YAML_DEFAULTS.get(
        "database_url_sync",
        "postgresql://raguser:ragpassword@localhost:5432/ragdb",
    )
    db_pool_size: int = _YAML_DEFAULTS.get("db_pool_size", 10)
    db_max_overflow: int = _YAML_DEFAULTS.get("db_max_overflow", 20)
    db_pool_timeout: int = _YAML_DEFAULTS.get("db_pool_timeout", 30)

    # ── Gemini (LLM) ─────────────────────────────────────────────
    gemini_api_key: Optional[str] = None
    llm_provider: str = _YAML_DEFAULTS.get("llm_provider", "gemini")
    llm_model: str = _YAML_DEFAULTS.get("llm_model", "gemini-1.5-flash")
    llm_max_tokens: int = _YAML_DEFAULTS.get("llm_max_tokens", 1024)
    llm_temperature: float = _YAML_DEFAULTS.get("llm_temperature", 0.1)

    # ── Embedding (Sentence Transformers) ────────────────────────
    # Provider: sentence_transformer | huggingface (alias) | gemini
    embedding_provider: str = _YAML_DEFAULTS.get("embedding_provider", "sentence_transformer")
    embedding_model: str = _YAML_DEFAULTS.get("embedding_model", "all-MiniLM-L6-v2")
    embedding_dimension: int = _YAML_DEFAULTS.get("embedding_dimension", 384)

    # Sentence Transformer runtime options
    st_batch_size: int = _YAML_DEFAULTS.get("st_batch_size", 64)
    st_device: str = _YAML_DEFAULTS.get("st_device", "cpu")   # cpu | cuda | mps
    st_normalize: bool = _YAML_DEFAULTS.get("st_normalize", True)

    # ── Vector Store (Qdrant) ────────────────────────────────────
    qdrant_url: str = _YAML_DEFAULTS.get("qdrant_url", "http://localhost:6333")
    qdrant_api_key: Optional[str] = None          # override via QDRANT_API_KEY
    qdrant_collection_name: str = _YAML_DEFAULTS.get(
        "qdrant_collection_name", "rag_knowledge_base"
    )
    qdrant_distance_metric: str = _YAML_DEFAULTS.get("qdrant_distance_metric", "cosine")
    qdrant_grpc_port: Optional[int] = None        # override via QDRANT_GRPC_PORT
    qdrant_on_disk_vectors: bool = _YAML_DEFAULTS.get("qdrant_on_disk_vectors", False)

    # ── File Processing ───────────────────────────────────────────
    max_file_size_mb: int = 50
    allowed_extensions: str = "pdf,py,txt,md,js,ts,java,cpp,c,go,rs"
    upload_dir: str = "./uploads"

    # ── Chunking ─────────────────────────────────────────────────
    chunk_size: int = _YAML_DEFAULTS.get("chunk_size", 512)
    chunk_overlap: int = _YAML_DEFAULTS.get("chunk_overlap", 50)
    min_chunk_size: int = _YAML_DEFAULTS.get("min_chunk_size", 100)

    # ── Query ────────────────────────────────────────────────────
    default_top_k: int = _YAML_DEFAULTS.get("default_top_k", 5)
    max_top_k: int = _YAML_DEFAULTS.get("max_top_k", 20)

    # ── Rate Limiting ─────────────────────────────────────────────
    rate_limit_requests: int = 100
    rate_limit_window: int = 60  # seconds

    # ── Logging ──────────────────────────────────────────────────
    log_level: str = "INFO"
    log_format: str = "json"

    # ── Derived helpers ───────────────────────────────────────────
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @field_validator("embedding_provider")
    @classmethod
    def validate_embedding_provider(cls, v: str) -> str:
        allowed = {"sentence_transformer", "huggingface", "gemini"}
        if v.lower() not in allowed:
            raise ValueError(f"embedding_provider must be one of {allowed}")
        return v.lower()

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        allowed = {"gemini"}
        if v.lower() not in allowed:
            raise ValueError(f"llm_provider must be one of {allowed}")
        return v.lower()


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance — call this everywhere."""
    return Settings()


settings = get_settings()

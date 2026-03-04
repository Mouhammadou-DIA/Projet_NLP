"""
Configuration Settings - Professional Reddit RAG Chatbot
Centralized configuration management with environment variables
"""

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Application settings with environment variable support

    All settings can be overridden via environment variables
    Example: export EMBEDDING_MODEL="paraphrase-multilingual-mpnet-base-v2"
    """

    # ==================== APPLICATION ====================
    APP_NAME: str = "Reddit RAG Chatbot"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "Professional multilingual chatbot using RAG architecture"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "production")  # development, staging, production
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"

    # ==================== PATHS ====================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DATA_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
    VECTOR_DB_DIR: Path = DATA_DIR / "vector_db"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # ==================== EMBEDDINGS ====================
    EMBEDDING_MODEL: str = "paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION: int = 384
    EMBEDDING_BATCH_SIZE: int = 500
    EMBEDDING_DEVICE: str = "cpu"  # or "cuda" for GPU

    # ==================== VECTOR STORE ====================
    VECTOR_STORE_TYPE: str = "chromadb"  # chromadb, pinecone, qdrant
    CHROMA_COLLECTION_NAME: str = "reddit_conversations_pro"
    CHROMA_PERSIST_DIRECTORY: str = str(VECTOR_DB_DIR / "chroma_db")

    # ==================== LLM ====================
    LLM_PROVIDER: str = "ollama"  # ollama, openai, anthropic, groq
    LLM_MODEL: str = "llama3.1:8b"  # Meta Llama 3.1 8B - optimal pour 16GB RAM
    LLM_TEMPERATURE: float = 0.7
    LLM_MAX_TOKENS: int = 500
    LLM_TIMEOUT: int = 60  # seconds (increased for larger model)
    OLLAMA_BASE_URL: str = "http://localhost:11434"  # Ollama server URL
    GROQ_API_KEY: str | None = None  # Groq API key (free at console.groq.com)

    # ==================== RERANKER ====================
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RERANKER_DEVICE: str = "cpu"
    RERANKER_TOP_K: int = 3  # Final number of results after reranking

    # ==================== CHATBOT ====================
    DEFAULT_N_RESULTS: int = 5
    MIN_SIMILARITY_SCORE: float = 0.5
    MAX_CONTEXT_LENGTH: int = 2000
    ENABLE_CACHING: bool = True
    CACHE_BACKEND: str = "memory"  # memory or redis
    CACHE_TTL: int = 3600  # seconds
    REDIS_URL: str = "redis://localhost:6379/0"

    # ==================== CONVERSATION MEMORY ====================
    MEMORY_MAX_SESSIONS: int = 1000
    MEMORY_SESSION_TIMEOUT: int = 3600  # 1 hour
    MEMORY_MAX_MESSAGES: int = 100
    MEMORY_SUMMARY_THRESHOLD: int = 10
    MEMORY_KEEP_RECENT: int = 5

    # ==================== API ====================
    API_HOST: str = "127.0.0.1"
    API_PORT: int = 8000
    API_WORKERS: int = 4
    API_RELOAD: bool = DEBUG
    API_RATE_LIMIT: str = "100/minute"
    CORS_ORIGINS: list = ["*"]  # Allow all origins in development

    # ==================== UI ====================
    UI_HOST: str = "127.0.0.1"
    UI_PORT: int = 3000
    UI_SHARE: bool = False
    UI_THEME: str = "soft"

    # ==================== LOGGING ====================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "json"  # json or text
    LOG_FILE: str | None = str(LOGS_DIR / "app.log")
    LOG_ROTATION: str = "100 MB"
    LOG_RETENTION: str = "30 days"

    # ==================== MONITORING ====================
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    SENTRY_DSN: str | None = os.getenv("SENTRY_DSN")

    # ==================== DATABASE ====================
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30

    # ==================== SECURITY ====================
    SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-in-production-please")
    API_KEY_HEADER: str = "X-API-Key"
    ENABLE_AUTH: bool = ENVIRONMENT == "production"

    # ==================== PERFORMANCE ====================
    ENABLE_ASYNC: bool = True
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: int = 30

    class Config:
        """Pydantic config"""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "ignore"  # Ignore extra env vars not defined in Settings

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create directories if they don't exist
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.VECTOR_DB_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance

    Returns:
        Settings: Singleton settings instance
    """
    return Settings()


# Export singleton instance
settings = get_settings()


# Convenience functions
def is_production() -> bool:
    """Check if running in production"""
    return settings.ENVIRONMENT == "production"


def is_development() -> bool:
    """Check if running in development"""
    return settings.ENVIRONMENT == "development"


def get_db_path() -> Path:
    """Get vector database path"""
    return Path(settings.CHROMA_PERSIST_DIRECTORY)

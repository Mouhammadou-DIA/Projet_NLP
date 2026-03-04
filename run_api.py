"""
Run API Server - Reddit RAG Chatbot
Entry point for running the FastAPI server
"""

import os

import uvicorn

from src.config.logging_config import log_startup
from src.config.settings import settings


if __name__ == "__main__":
    log_startup()

    port = int(os.getenv("PORT", settings.API_PORT))
    host = settings.API_HOST

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        workers=settings.API_WORKERS if not settings.DEBUG else 1,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )

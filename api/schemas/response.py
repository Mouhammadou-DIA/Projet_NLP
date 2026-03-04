"""
API Response Schemas - Professional Reddit RAG Chatbot
Pydantic schemas for API responses
"""

from typing import Any

from pydantic import BaseModel, Field

from src.models.schemas import SearchResult


class ChatResponseAPI(BaseModel):
    """
    Chat response schema for API

    Extended version of ChatResponse with API-specific fields
    """

    message: str = Field(..., description="Generated response")
    sources: list[SearchResult] = Field(
        default_factory=list, description="Source conversations used"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Response metadata")

    class Config:
        schema_extra = {
            "example": {
                "message": "I recommend the Pixel phone. It's fast and has great camera.",
                "sources": [],
                "metadata": {"duration_ms": 234.56, "method": "simple", "n_sources": 5},
            }
        }


class SearchResponseAPI(BaseModel):
    """Response schema for search results"""

    results: list[SearchResult] = Field(..., description="Search results")
    total: int = Field(..., description="Total number of results")
    query: str = Field(..., description="Original query")

    class Config:
        schema_extra = {"example": {"results": [], "total": 5, "query": "What phone should I buy?"}}


class StatsResponse(BaseModel):
    """Response schema for statistics"""

    total_conversations: int
    embedding_model: str
    llm_model: str
    llm_available: bool
    cache_enabled: bool

    class Config:
        schema_extra = {
            "example": {
                "total_conversations": 56295,
                "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
                "llm_model": "llama3.2",
                "llm_available": True,
                "cache_enabled": True,
            }
        }


class VersionResponse(BaseModel):
    """Response schema for version information"""

    app: str
    version: str
    environment: str
    python_version: str
    models: dict[str, str]

    class Config:
        schema_extra = {
            "example": {
                "app": "Reddit RAG Chatbot",
                "version": "2.0.0",
                "environment": "production",
                "python_version": "3.12",
                "models": {"embedding": "paraphrase-multilingual-MiniLM-L12-v2", "llm": "llama3.2"},
            }
        }

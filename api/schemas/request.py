"""
API Request Schemas - Professional Reddit RAG Chatbot
Pydantic schemas for API requests
"""

from pydantic import BaseModel, Field

from src.models.schemas import ChatMessage


class ChatRequestAPI(BaseModel):
    """
    Chat request schema for API

    Extended version of ChatRequest for API endpoints
    """

    message: str = Field(..., min_length=1, max_length=1000, description="User message")
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, description="Previous conversation messages"
    )
    use_llm: bool = Field(default=True, description="Use LLM for generation")
    n_results: int = Field(
        default=5, ge=1, le=20, description="Number of similar conversations to retrieve"
    )
    temperature: float | None = Field(default=0.7, ge=0, le=2, description="LLM temperature")
    max_tokens: int | None = Field(
        default=500, ge=1, le=2000, description="Maximum tokens to generate"
    )

    class Config:
        schema_extra = {
            "example": {
                "message": "What phone should I buy?",
                "session_id": None,
                "use_llm": True,
                "n_results": 5,
            }
        }


class SearchRequest(BaseModel):
    """Request schema for searching similar conversations"""

    query: str = Field(..., min_length=1, max_length=500, description="Search query")
    n_results: int = Field(default=5, ge=1, le=20, description="Number of results")
    min_score: float = Field(default=0.0, ge=0, le=1, description="Minimum similarity score")

    class Config:
        schema_extra = {
            "example": {"query": "How do I make friends?", "n_results": 10, "min_score": 0.5}
        }

"""
Data Models - Professional Reddit RAG Chatbot
Pydantic models for type safety and validation
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, validator


class MessageRole(str, Enum):
    """Message role enum"""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """
    Chat message model

    Attributes:
        role: Message role (user/assistant/system)
        content: Message content
        timestamp: Message timestamp
        metadata: Additional metadata
    """

    role: MessageRole
    content: str = Field(..., min_length=1, max_length=10000)
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: dict[str, Any] | None = None

    @validator("content")
    def content_not_empty(cls, v):
        """Validate content is not just whitespace"""
        if not v.strip():
            raise ValueError("Content cannot be empty or whitespace only")
        return v.strip()

    class Config:
        """Pydantic config"""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class Conversation(BaseModel):
    """
    Conversation model (from Reddit data)

    Attributes:
        id: Unique conversation ID
        context: Initial message or context
        response: Response to context
        follow_up: Optional follow-up message
        full_text: Combined text for embedding
        embedding: Vector embedding (optional)
        metadata: Additional metadata
    """

    id: int = Field(..., ge=0)
    context: str = Field(..., min_length=1)
    response: str = Field(..., min_length=1)
    follow_up: str | None = None
    full_text: str | None = None
    embedding: list[float] | None = None
    metadata: dict[str, Any] | None = Field(default_factory=dict)

    def __init__(self, **data):
        """Initialize and auto-generate full_text"""
        super().__init__(**data)
        if not self.full_text:
            self.full_text = f"Question: {self.context}\nRéponse: {self.response}"

    @validator("context", "response")
    def text_not_empty(cls, v):
        """Validate text fields are not empty"""
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

    class Config:
        """Pydantic config"""

        validate_assignment = True


class SearchResult(BaseModel):
    """
    Search result model

    Attributes:
        conversation: Matched conversation
        score: Relevance score (can be negative for cross-encoders)
        distance: Distance metric
        rank: Result rank
    """

    conversation: Conversation
    score: float = Field(..., description="Relevance score (not necessarily normalized)")
    distance: float | None = None
    rank: int = Field(..., ge=1)

    class Config:
        """Pydantic config"""

        arbitrary_types_allowed = True


class ChatRequest(BaseModel):
    """
    Chat request model

    Attributes:
        message: User message
        conversation_history: Previous messages
        use_llm: Whether to use LLM for generation
        n_results: Number of similar conversations to retrieve
        temperature: LLM temperature (if using LLM)
        max_tokens: Maximum tokens to generate
    """

    message: str = Field(..., min_length=1, max_length=1000)
    session_id: str | None = Field(
        default=None, description="Session ID for conversation continuity"
    )
    conversation_history: list[ChatMessage] = Field(default_factory=list)
    use_llm: bool = False
    n_results: int = Field(default=5, ge=1, le=20)
    temperature: float | None = Field(default=0.7, ge=0, le=2)
    max_tokens: int | None = Field(default=500, ge=1, le=2000)

    @validator("message")
    def message_not_empty(cls, v):
        """Validate message is not empty"""
        if not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()

    @validator("conversation_history")
    def history_not_too_long(cls, v):
        """Validate conversation history length"""
        if len(v) > 50:
            raise ValueError("Conversation history too long (max 50 messages)")
        return v


class ChatResponse(BaseModel):
    """
    Chat response model

    Attributes:
        message: Assistant's response
        sources: Source conversations used
        metadata: Response metadata (timing, model, etc.)
    """

    message: str
    sources: list[SearchResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic config"""

        arbitrary_types_allowed = True


class HealthStatus(str, Enum):
    """Health status enum"""

    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"


class HealthCheck(BaseModel):
    """
    Health check response

    Attributes:
        status: Overall health status
        version: Application version
        components: Component health statuses
        timestamp: Check timestamp
    """

    status: HealthStatus
    version: str
    components: dict[str, dict[str, Any]]
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config"""

        use_enum_values = True
        json_encoders = {datetime: lambda v: v.isoformat()}


class ErrorResponse(BaseModel):
    """
    Error response model

    Attributes:
        error: Error message
        detail: Detailed error information
        code: Error code
        timestamp: Error timestamp
    """

    error: str
    detail: str | None = None
    code: str | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        """Pydantic config"""

        json_encoders = {datetime: lambda v: v.isoformat()}


class IndexingStatus(BaseModel):
    """
    Indexing status model

    Attributes:
        total_documents: Total documents to index
        indexed_documents: Documents already indexed
        progress: Progress percentage
        status: Current status
        errors: List of errors encountered
    """

    total_documents: int
    indexed_documents: int
    progress: float = Field(..., ge=0, le=100)
    status: str
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    class Config:
        """Pydantic config"""

        json_encoders = {datetime: lambda v: v.isoformat() if v else None}

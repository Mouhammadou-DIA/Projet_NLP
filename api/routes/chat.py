"""
Chat API Routes - Professional Reddit RAG Chatbot
REST endpoints for chat functionality
"""

from fastapi import APIRouter, HTTPException, status

from src.config.logging_config import get_logger
from src.models.schemas import ChatRequest, ChatResponse, ErrorResponse
from src.services.chatbot_service import get_chatbot_service


logger = get_logger(__name__)
router = APIRouter()


@router.post(
    "/",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    summary="Send a chat message",
    description="""
    Send a message to the chatbot and receive a response.

    The chatbot uses RAG (Retrieval-Augmented Generation) to find relevant
    conversations from Reddit and generate an appropriate response.

    **Supports:**
    - Questions in French or English
    - Simple mode (fast, retrieval-only)
    - LLM mode (slower, generates natural responses)
    """,
    responses={
        200: {"description": "Successful response"},
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    """
    Chat endpoint

    Args:
        request: Chat request with message and parameters

    Returns:
        ChatResponse: Bot response with sources and metadata

    Raises:
        HTTPException: If request fails
    """
    try:
        logger.info(f"Chat request received: {request.message[:50]}...")

        # Get chatbot service
        chatbot = get_chatbot_service()

        # Process request with session_id for conversation continuity
        response = chatbot.chat(request, session_id=request.session_id)

        logger.info(f"Chat response generated ({len(response.message)} chars)")
        return response

    except ValueError as e:
        logger.warning(f"Invalid request: {e!s}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.error(f"Chat failed: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process chat request",
        )


@router.get(
    "/stats",
    summary="Get chatbot statistics",
    description="Get statistics about the chatbot (total conversations, models, etc.)",
)
async def get_stats():
    """
    Get chatbot statistics

    Returns:
        Statistics dictionary
    """
    try:
        chatbot = get_chatbot_service()
        stats = chatbot.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Failed to get stats: {e!s}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve statistics",
        )


@router.get(
    "/examples",
    summary="Get example questions",
    description="Get a list of example questions to ask the chatbot",
)
async def get_examples():
    """
    Get example questions

    Returns:
        List of example questions in French and English
    """
    return {
        "french": [
            "Quel téléphone me recommandes-tu ?",
            "Je me sens triste aujourd'hui",
            "Comment faire des amis ?",
            "Je viens d'avoir un nouveau travail",
            "Je me marie bientôt",
        ],
        "english": [
            "What phone should I buy?",
            "I'm feeling sad today",
            "How do I make friends?",
            "I just got a new job",
            "I'm getting married soon",
        ],
    }

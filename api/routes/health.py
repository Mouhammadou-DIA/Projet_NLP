"""
Health Check API Routes - Professional Reddit RAG Chatbot
Endpoints for monitoring and health checks
"""

from datetime import datetime

from fastapi import APIRouter, status

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.models.schemas import HealthCheck, HealthStatus
from src.services.chatbot_service import get_chatbot_service


logger = get_logger(__name__)
router = APIRouter()


@router.get(
    "/",
    response_model=HealthCheck,
    status_code=status.HTTP_200_OK,
    summary="Health check",
    description="Check if the application is healthy and running",
)
async def health_check() -> HealthCheck:
    """
    Comprehensive health check

    Returns:
        HealthCheck: Health status of all components
    """
    try:
        # Get chatbot service
        chatbot = get_chatbot_service()

        # Check components
        component_health = chatbot.health_check()

        # Determine overall status
        unhealthy_components = [
            name for name, status_str in component_health.items() if status_str == "unhealthy"
        ]

        if unhealthy_components:
            overall_status = HealthStatus.UNHEALTHY
        elif any(s == "unavailable" for s in component_health.values()):
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY

        # Build detailed component info
        components = {
            name: {"status": status_str, "message": _get_component_message(name, status_str)}
            for name, status_str in component_health.items()
        }

        return HealthCheck(
            status=overall_status,
            version=settings.APP_VERSION,
            components=components,
            timestamp=datetime.utcnow(),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e!s}")
        return HealthCheck(
            status=HealthStatus.UNHEALTHY,
            version=settings.APP_VERSION,
            components={"error": {"status": "unhealthy", "message": str(e)}},
            timestamp=datetime.utcnow(),
        )


@router.get(
    "/ready",
    status_code=status.HTTP_200_OK,
    summary="Readiness check",
    description="Check if the application is ready to serve requests",
)
async def readiness_check():
    """
    Kubernetes-style readiness probe

    Returns:
        Simple ready/not ready status
    """
    try:
        chatbot = get_chatbot_service()
        health = chatbot.health_check()

        # Check if critical components are healthy
        if health.get("vector_store") == "unhealthy":
            return {"ready": False, "reason": "Vector store unavailable"}

        if health.get("embedding_service") == "unhealthy":
            return {"ready": False, "reason": "Embedding service unavailable"}

        return {"ready": True}

    except Exception as e:
        logger.error(f"Readiness check failed: {e!s}")
        return {"ready": False, "reason": str(e)}


@router.get(
    "/live",
    status_code=status.HTTP_200_OK,
    summary="Liveness check",
    description="Check if the application is alive (for container orchestration)",
)
async def liveness_check():
    """
    Kubernetes-style liveness probe

    Returns:
        Simple alive status
    """
    return {"alive": True, "timestamp": datetime.utcnow().isoformat()}


@router.get(
    "/version",
    status_code=status.HTTP_200_OK,
    summary="Get version info",
    description="Get application version and build information",
)
async def version_info():
    """
    Get version information

    Returns:
        Version details
    """
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "python_version": "3.12",
        "models": {
            "embedding": settings.EMBEDDING_MODEL,
            "llm": settings.LLM_MODEL,
        },
    }


def _get_component_message(name: str, status: str) -> str:
    """
    Get human-readable message for component status

    Args:
        name: Component name
        status: Component status

    Returns:
        Status message
    """
    messages = {
        ("embedding_service", "healthy"): "Embedding service operational",
        ("embedding_service", "unhealthy"): "Embedding service down",
        ("vector_store", "healthy"): "Vector store operational",
        ("vector_store", "unhealthy"): "Vector store unavailable",
        ("llm_service", "healthy"): "LLM service operational",
        ("llm_service", "unavailable"): "LLM service not configured (optional)",
        ("llm_service", "unhealthy"): "LLM service down",
    }

    return messages.get((name, status), f"{name} is {status}")

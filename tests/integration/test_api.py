"""
Integration tests for the FastAPI application.
"""

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def mock_chatbot_service():
    """Create mock ChatbotService."""
    mock_service = MagicMock()
    mock_service.chat.return_value = MagicMock(
        message="This is a test response.",
        sources=[],
        metadata={
            "duration_ms": 100,
            "method": "simple",
            "n_sources": 1,
            "model": None,
        },
        dict=lambda: {
            "message": "This is a test response.",
            "sources": [],
            "metadata": {
                "duration_ms": 100,
                "method": "simple",
                "n_sources": 1,
                "model": None,
            },
        },
    )
    mock_service.health_check.return_value = {
        "status": "healthy",
        "embedding_service": "healthy",
        "vector_store": "healthy",
        "llm_service": "healthy",
    }
    mock_service.get_stats.return_value = {
        "total_conversations": 56295,
        "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        "llm_provider": "ollama",
    }
    return mock_service


@pytest.fixture
def client(mock_chatbot_service):
    """Create test client with mocked service."""
    with (
        patch("api.routes.chat.get_chatbot_service", return_value=mock_chatbot_service),
        patch("api.routes.health.get_chatbot_service", return_value=mock_chatbot_service),
    ):
        from fastapi.testclient import TestClient

        from api.main import app

        with TestClient(app) as client:
            yield client


class TestRootEndpoint:
    """Tests for root endpoint."""

    @pytest.mark.integration
    def test_root_returns_200(self, client):
        """Test root endpoint returns 200."""
        response = client.get("/api")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_root_returns_app_info(self, client):
        """Test root endpoint returns application info."""
        response = client.get("/api")
        data = response.json()
        assert "app" in data or "status" in data


class TestChatEndpoint:
    """Tests for chat endpoint."""

    @pytest.mark.integration
    def test_chat_endpoint_success(self, client):
        """Test chat endpoint with valid request."""
        response = client.post("/api/v1/chat/", json={"message": "Hello, how are you?"})
        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_endpoint_empty_message(self, client):
        """Test chat endpoint with empty message."""
        response = client.post("/api/v1/chat/", json={"message": ""})
        assert response.status_code in [400, 422]

    @pytest.mark.integration
    def test_chat_endpoint_missing_message(self, client):
        """Test chat endpoint with missing message field."""
        response = client.post("/api/v1/chat/", json={})
        assert response.status_code == 422

    @pytest.mark.integration
    def test_chat_stats_endpoint(self, client):
        """Test chat stats endpoint."""
        response = client.get("/api/v1/chat/stats")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_chat_examples_endpoint(self, client):
        """Test chat examples endpoint."""
        response = client.get("/api/v1/chat/examples")
        assert response.status_code == 200


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.mark.integration
    def test_health_endpoint(self, client):
        """Test main health endpoint."""
        response = client.get("/api/v1/health/")
        assert response.status_code == 200

    @pytest.mark.integration
    def test_health_live_endpoint(self, client):
        """Test liveness probe endpoint."""
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200


class TestAPIDocumentation:
    """Tests for API documentation endpoints."""

    @pytest.mark.integration
    def test_openapi_json(self, client):
        """Test OpenAPI JSON endpoint (may be disabled in non-debug mode)."""
        response = client.get("/openapi.json")
        # Docs are disabled when DEBUG=False, so 404 is acceptable
        assert response.status_code in [200, 404]

    @pytest.mark.integration
    def test_docs_endpoint(self, client):
        """Test Swagger UI endpoint (may be disabled in non-debug mode)."""
        response = client.get("/docs")
        assert response.status_code in [200, 404]


class TestErrorHandling:
    """Tests for error handling."""

    @pytest.mark.integration
    def test_404_not_found(self, client):
        """Test 404 for non-existent endpoint."""
        response = client.get("/api/v1/nonexistent")
        assert response.status_code == 404

    @pytest.mark.integration
    def test_invalid_json(self, client):
        """Test handling of invalid JSON."""
        response = client.post(
            "/api/v1/chat/", content="invalid json", headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422

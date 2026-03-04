"""
Pytest configuration and shared fixtures.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Environment Fixtures
# =============================================================================


@pytest.fixture(scope="session", autouse=True)
def set_test_environment():
    """Set test environment variables."""
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DEBUG"] = "false"
    os.environ["LOG_LEVEL"] = "WARNING"


# =============================================================================
# Sample Data Fixtures
# =============================================================================


@pytest.fixture
def sample_conversations() -> list[dict]:
    """Load sample conversations from fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures" / "sample_conversations.json"
    if fixtures_path.exists():
        with open(fixtures_path) as f:
            return json.load(f)
    return [
        {
            "id": 1,
            "context": "What is machine learning?",
            "response": "Machine learning is a subset of AI that enables systems to learn from data.",
            "subreddit": "test",
        },
        {
            "id": 2,
            "context": "How does Python work?",
            "response": "Python is an interpreted language that executes code line by line.",
            "subreddit": "test",
        },
        {
            "id": 3,
            "context": "What is the meaning of life?",
            "response": "That's a philosophical question with many different answers.",
            "subreddit": "test",
        },
    ]


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for embedding tests."""
    return [
        "Hello, how are you today?",
        "What is artificial intelligence?",
        "Tell me about machine learning.",
        "How does natural language processing work?",
        "Explain deep learning concepts.",
    ]


@pytest.fixture
def sample_query() -> str:
    """Sample query for search tests."""
    return "What is machine learning and how does it work?"


# =============================================================================
# Mock Fixtures
# =============================================================================


@pytest.fixture
def mock_embedding_model():
    """Mock sentence-transformers model."""
    mock_model = MagicMock()
    mock_model.encode.return_value = [[0.1] * 384]
    mock_model.get_sentence_embedding_dimension.return_value = 384
    return mock_model


@pytest.fixture
def mock_embeddings() -> list[list[float]]:
    """Mock embedding vectors."""
    import random

    return [[random.random() for _ in range(384)] for _ in range(5)]


@pytest.fixture
def mock_chroma_client():
    """Mock ChromaDB client."""
    mock_client = MagicMock()
    mock_collection = MagicMock()
    mock_collection.count.return_value = 100
    mock_collection.query.return_value = {
        "ids": [["conv_1", "conv_2"]],
        "documents": [["doc1", "doc2"]],
        "metadatas": [[{"context": "q1", "response": "a1"}, {"context": "q2", "response": "a2"}]],
        "distances": [[0.1, 0.2]],
    }
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client


@pytest.fixture
def mock_llm_response() -> str:
    """Mock LLM response."""
    return "This is a generated response based on the provided context."


# =============================================================================
# Service Fixtures
# =============================================================================


@pytest.fixture
def embedding_service():
    """Create EmbeddingService with mocked model."""
    with patch("src.core.embeddings.SentenceTransformer") as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1] * 384]
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()
        yield service


@pytest.fixture
def vector_store_service(mock_chroma_client):
    """Create VectorStoreService with mocked ChromaDB."""
    with patch("src.core.vector_store.chromadb") as mock_chromadb:
        mock_chromadb.PersistentClient.return_value = mock_chroma_client

        from src.core.vector_store import VectorStoreService

        service = VectorStoreService()
        yield service


@pytest.fixture
def llm_service():
    """Create LLMService with mocked provider."""
    with patch("src.core.llm_handler.ollama") as mock_ollama:
        mock_ollama.chat.return_value = {"message": {"content": "This is a mock LLM response."}}

        from src.core.llm_handler import LLMService

        service = LLMService()
        yield service


@pytest.fixture
def chatbot_service(embedding_service, vector_store_service, llm_service):
    """Create ChatbotService with mocked dependencies."""
    with (
        patch("src.services.chatbot_service.EmbeddingService", return_value=embedding_service),
        patch("src.services.chatbot_service.VectorStoreService", return_value=vector_store_service),
        patch("src.services.chatbot_service.LLMService", return_value=llm_service),
    ):
        from src.services.chatbot_service import ChatbotService

        service = ChatbotService()
        yield service


# =============================================================================
# API Fixtures
# =============================================================================


@pytest.fixture
def test_client():
    """Create FastAPI test client."""
    from fastapi.testclient import TestClient

    with patch("api.main.ChatbotService") as mock_service:
        mock_instance = MagicMock()
        mock_instance.chat.return_value = {
            "message": "Test response",
            "sources": [],
            "metadata": {"duration_ms": 100, "mode": "simple"},
        }
        mock_instance.health_check.return_value = {
            "status": "healthy",
            "components": {},
        }
        mock_instance.get_stats.return_value = {
            "total_conversations": 100,
        }
        mock_service.return_value = mock_instance

        from api.main import app

        client = TestClient(app)
        yield client


# =============================================================================
# Async Fixtures
# =============================================================================


@pytest.fixture
def event_loop():
    """Create event loop for async tests."""
    import asyncio

    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Temporary Files/Directories
# =============================================================================


@pytest.fixture
def temp_data_dir(tmp_path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "raw").mkdir()
    (data_dir / "processed").mkdir()
    (data_dir / "vector_db").mkdir()
    return data_dir


@pytest.fixture
def temp_conversations_file(temp_data_dir, sample_conversations) -> Path:
    """Create temporary conversations JSON file."""
    file_path = temp_data_dir / "processed" / "conversations.json"
    with open(file_path, "w") as f:
        json.dump(sample_conversations, f)
    return file_path


# =============================================================================
# Markers
# =============================================================================


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")

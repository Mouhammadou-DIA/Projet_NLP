"""
Unit Tests - Chatbot Service
"""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from src.models.schemas import ChatRequest, Conversation, SearchResult


class TestChatbotService:
    """Test suite for ChatbotService"""

    @pytest.fixture
    def mock_services(self):
        """Create mock services"""
        embedding_service = Mock()
        vector_store = Mock()
        llm_service = Mock()
        cache_service = Mock()
        memory = Mock()

        # Configure defaults
        llm_service.is_available.return_value = False
        cache_service.enabled = True
        cache_service.get.return_value = None
        cache_service.set.return_value = None
        cache_service.get_stats.return_value = {}
        memory.get_or_create_session.return_value = Mock(session_id="test-session")
        memory.add_message.return_value = None
        memory.get_stats.return_value = {}

        return embedding_service, vector_store, llm_service, cache_service, memory

    @pytest.fixture
    def chatbot_service(self, mock_services):
        """Create chatbot service with mocks"""
        embedding_service, vector_store, llm_service, cache_service, memory = mock_services

        with patch("src.services.chatbot_service.SummarizingMemory") as mock_summarizing_cls:
            mock_summarizing = Mock()
            mock_summarizing.get_context.return_value = ""
            mock_summarizing_cls.return_value = mock_summarizing

            from src.services.chatbot_service import ChatbotService

            service = ChatbotService(
                embedding_service=embedding_service,
                vector_store=vector_store,
                llm_service=llm_service,
                reranker=None,
                cache_service=cache_service,
                conversation_memory=memory,
            )

        return service

    def test_initialization(self, chatbot_service):
        """Test service initialization"""
        assert chatbot_service is not None
        assert chatbot_service.embedding_service is not None
        assert chatbot_service.vector_store is not None
        assert chatbot_service.llm_service is not None

    def test_chat_simple_mode(self, chatbot_service, mock_services):
        """Test chat in simple mode"""
        embedding_service, vector_store, _llm_service, _cache, _memory = mock_services

        embedding_service.embed_text.return_value = np.array([0.1, 0.2, 0.3])

        conv = Conversation(id=1, context="What phone?", response="I recommend Pixel")
        search_result = SearchResult(conversation=conv, score=0.95, rank=1)
        vector_store.search.return_value = [search_result]

        request = ChatRequest(message="What phone should I buy?", use_llm=False)
        response = chatbot_service.chat(request)

        assert response is not None
        assert response.message == "I recommend Pixel"
        assert len(response.sources) > 0
        embedding_service.embed_text.assert_called_once()
        vector_store.search.assert_called_once()

    def test_chat_with_llm(self, chatbot_service, mock_services):
        """Test chat with LLM mode"""
        embedding_service, vector_store, llm_service, _cache, _memory = mock_services

        embedding_service.embed_text.return_value = np.array([0.1, 0.2, 0.3])

        conv = Conversation(id=1, context="What phone?", response="I recommend Pixel")
        vector_store.search.return_value = [SearchResult(conversation=conv, score=0.95, rank=1)]

        llm_service.is_available.return_value = True
        llm_service.generate.return_value = "Based on the context, I recommend a Pixel phone."

        request = ChatRequest(message="What phone should I buy?", use_llm=True)
        response = chatbot_service.chat(request)

        assert response is not None
        assert "Pixel" in response.message
        llm_service.generate.assert_called_once()

    def test_chat_empty_message(self):
        """Test chat with empty message raises validation error"""
        with pytest.raises(ValidationError):
            ChatRequest(message="   ", use_llm=False)

    def test_chat_no_results(self, chatbot_service, mock_services):
        """Test chat when no similar conversations found"""
        embedding_service, vector_store, _llm_service, _cache, _memory = mock_services

        embedding_service.embed_text.return_value = np.array([0.1, 0.2, 0.3])
        vector_store.search.return_value = []

        request = ChatRequest(message="What phone should I buy?", use_llm=False)
        response = chatbot_service.chat(request)

        assert response is not None
        assert (
            "couldn't find" in response.message.lower() or "no relevant" in response.message.lower()
        )

    def test_get_stats(self, chatbot_service, mock_services):
        """Test getting statistics"""
        _embedding_service, vector_store, llm_service, _cache, _memory = mock_services

        vector_store.count.return_value = 56295
        llm_service.is_available.return_value = True

        stats = chatbot_service.get_stats()

        assert stats is not None
        assert "total_conversations" in stats
        assert stats["total_conversations"] == 56295
        assert "llm_available" in stats

    def test_health_check(self, chatbot_service, mock_services):
        """Test health check"""
        embedding_service, vector_store, llm_service, _cache, _memory = mock_services

        embedding_service.embed_text.return_value = np.array([0.1])
        vector_store.count.return_value = 100
        llm_service.is_available.return_value = True

        health = chatbot_service.health_check()

        assert health is not None
        assert "embedding_service" in health
        assert "vector_store" in health
        assert "llm_service" in health


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

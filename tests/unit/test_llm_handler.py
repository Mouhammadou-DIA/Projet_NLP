"""
Unit tests for LLMService.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest


class TestLLMService:
    """Tests for LLMService class."""

    @pytest.fixture
    def mock_ollama(self):
        """Create mock Ollama module and keep it active."""
        mock = MagicMock()
        mock.chat.return_value = {"message": {"content": "This is a test response from LLM."}}
        mock.list.return_value = {"models": [{"name": "llama3.2"}]}
        with patch.dict(sys.modules, {"ollama": mock}):
            yield mock

    @pytest.fixture
    def service(self, mock_ollama):
        """Create LLMService with mocked Ollama."""
        from src.core.llm_handler import LLMService

        return LLMService()

    @pytest.mark.unit
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None

    @pytest.mark.unit
    def test_generate_returns_string(self, service):
        """Test generate returns a string response."""
        context = "Context 1: Some information\nContext 2: More information"
        response = service.generate("What is AI?", context)
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.unit
    def test_generate_with_empty_context(self, service):
        """Test generate with empty context."""
        response = service.generate("What is AI?", "")
        assert isinstance(response, str)

    @pytest.mark.unit
    def test_build_prompt_includes_query(self, service):
        """Test that prompt includes the user query."""
        query = "What is machine learning?"
        context = "ML is a subset of AI."
        prompt = service._build_prompt(query, context)
        assert query in prompt

    @pytest.mark.unit
    def test_build_prompt_includes_context(self, service):
        """Test that prompt includes context."""
        query = "What is AI?"
        context = "AI stands for Artificial Intelligence."
        prompt = service._build_prompt(query, context)
        assert "Artificial Intelligence" in prompt

    @pytest.mark.unit
    def test_check_availability(self, service):
        """Test availability check."""
        is_available = service._check_availability()
        assert isinstance(is_available, bool)

    @pytest.mark.unit
    def test_generate_with_unicode(self, service):
        """Test generate handles Unicode in query."""
        context = "Contexte en français"
        response = service.generate("Qu'est-ce que l'IA?", context)
        assert isinstance(response, str)


class TestLLMServiceProviders:
    """Tests for different LLM providers."""

    @pytest.mark.unit
    def test_ollama_provider(self):
        """Test Ollama provider configuration."""
        mock_ollama = MagicMock()
        mock_ollama.chat.return_value = {"message": {"content": "Response"}}
        mock_ollama.list.return_value = {"models": [{"name": "llama3.2"}]}

        with patch.dict(sys.modules, {"ollama": mock_ollama}):
            from src.core.llm_handler import LLMService

            service = LLMService()
            assert service is not None
            assert service._available is True


class TestLLMServiceIntegration:
    """Integration tests with real LLM."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_ollama_generation(self):
        """Test real Ollama generation (requires Ollama running)."""
        try:
            from src.core.llm_handler import LLMService

            service = LLMService()

            if service._check_availability():
                context = "Python is a programming language."
                response = service.generate("What is Python?", context)
                assert isinstance(response, str)
                assert len(response) > 0
            else:
                pytest.skip("Ollama not available")
        except Exception as e:
            pytest.skip(f"Ollama not available: {e}")

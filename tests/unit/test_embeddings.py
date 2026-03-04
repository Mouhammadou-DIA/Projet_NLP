"""
Unit tests for EmbeddingService.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestEmbeddingService:
    """Tests for EmbeddingService class."""

    @pytest.fixture
    def mock_sentence_transformer(self):
        """Create a mock SentenceTransformer model."""
        mock_model = MagicMock()
        mock_model.encode.return_value = np.array([0.1] * 384)
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.max_seq_length = 512
        return mock_model

    @pytest.fixture
    def service(self, mock_sentence_transformer):
        """Create EmbeddingService with mocked model."""
        with patch(
            "src.core.embeddings.SentenceTransformer", return_value=mock_sentence_transformer
        ):
            from src.core.embeddings import EmbeddingService

            return EmbeddingService()

    @pytest.mark.unit
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.model is not None

    @pytest.mark.unit
    def test_embed_text_returns_ndarray(self, service):
        """Test embed_text returns a numpy array."""
        embedding = service.embed_text("Hello world")
        assert isinstance(embedding, np.ndarray)
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_embed_text_correct_dimension(self, service):
        """Test embedding has correct dimension."""
        embedding = service.embed_text("Test text")
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_embed_batch_returns_ndarray(self, service, mock_sentence_transformer):
        """Test embed_batch returns ndarray of embeddings."""
        mock_sentence_transformer.encode.return_value = np.array([[0.1] * 384, [0.2] * 384])

        texts = ["Hello", "World"]
        embeddings = service.embed_batch(texts)

        assert isinstance(embeddings, np.ndarray)
        assert len(embeddings) == 2

    @pytest.mark.unit
    def test_get_similarity_identical_vectors(self, service):
        """Test similarity of identical vectors is 1.0."""
        vec = np.array([0.1] * 384)
        similarity = service.get_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, rel=1e-5)

    @pytest.mark.unit
    def test_get_similarity_orthogonal_vectors(self, service):
        """Test similarity of orthogonal vectors is 0.0."""
        vec1 = np.array([1.0] + [0.0] * 383)
        vec2 = np.array([0.0, 1.0] + [0.0] * 382)
        similarity = service.get_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-5)

    @pytest.mark.unit
    def test_get_similarity_opposite_vectors(self, service):
        """Test similarity of opposite vectors is -1.0."""
        vec1 = np.array([1.0] * 384)
        vec2 = np.array([-1.0] * 384)
        similarity = service.get_similarity(vec1, vec2)
        assert similarity == pytest.approx(-1.0, rel=1e-5)

    @pytest.mark.unit
    def test_get_similarity_range(self, service):
        """Test similarity is always between -1 and 1."""
        import random

        for _ in range(10):
            vec1 = np.array([random.random() for _ in range(384)])
            vec2 = np.array([random.random() for _ in range(384)])
            similarity = service.get_similarity(vec1, vec2)
            assert -1.0 <= similarity <= 1.0

    @pytest.mark.unit
    def test_unicode_text_handling(self, service):
        """Test Unicode text is handled correctly."""
        embedding = service.embed_text("Bonjour le monde!")
        assert len(embedding) == 384

    @pytest.mark.unit
    def test_get_model_info(self, service):
        """Test get_model_info returns dict."""
        info = service.get_model_info()
        assert isinstance(info, dict)
        assert "model_name" in info


class TestEmbeddingServiceIntegration:
    """Integration tests that require actual model loading."""

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_model_loading(self):
        """Test actual model loads correctly."""
        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()
        assert service.model is not None

    @pytest.mark.slow
    @pytest.mark.integration
    def test_semantic_similarity(self):
        """Test semantic similarity between related texts."""
        from src.core.embeddings import EmbeddingService

        service = EmbeddingService()

        emb1 = service.embed_text("I love programming in Python")
        emb2 = service.embed_text("Python is my favorite programming language")
        emb3 = service.embed_text("The weather is nice today")

        sim_related = service.get_similarity(emb1, emb2)
        sim_unrelated = service.get_similarity(emb1, emb3)

        assert sim_related > sim_unrelated

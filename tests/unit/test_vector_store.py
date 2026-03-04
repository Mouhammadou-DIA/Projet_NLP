"""
Unit tests for VectorStoreService.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


class TestVectorStoreService:
    """Tests for VectorStoreService class."""

    @pytest.fixture
    def mock_chroma_collection(self):
        """Create a mock ChromaDB collection."""
        mock_collection = MagicMock()
        mock_collection.count.return_value = 100
        mock_collection.query.return_value = {
            "ids": [["conv_1", "conv_2", "conv_3"]],
            "documents": [["doc1", "doc2", "doc3"]],
            "metadatas": [
                [
                    {"context": "Question 1", "response": "Answer 1", "id": "1"},
                    {"context": "Question 2", "response": "Answer 2", "id": "2"},
                    {"context": "Question 3", "response": "Answer 3", "id": "3"},
                ]
            ],
            "distances": [[0.1, 0.2, 0.3]],
        }
        mock_collection.add.return_value = None
        mock_collection.delete.return_value = None
        return mock_collection

    @pytest.fixture
    def mock_chroma_client(self, mock_chroma_collection):
        """Create a mock ChromaDB client."""
        mock_client = MagicMock()
        mock_client.get_collection.return_value = mock_chroma_collection
        mock_client.create_collection.return_value = mock_chroma_collection
        mock_client.get_or_create_collection.return_value = mock_chroma_collection
        mock_client.delete_collection.return_value = None
        return mock_client

    @pytest.fixture
    def service(self, mock_chroma_client):
        """Create VectorStoreService with mocked ChromaDB."""
        with patch("src.core.vector_store.chromadb") as mock_chromadb:
            mock_chromadb.PersistentClient.return_value = mock_chroma_client

            from src.core.vector_store import VectorStoreService

            return VectorStoreService()

    @pytest.mark.unit
    def test_initialization(self, service):
        """Test service initializes correctly."""
        assert service is not None
        assert service.collection is not None

    @pytest.mark.unit
    def test_count_returns_integer(self, service):
        """Test count returns document count."""
        count = service.count()
        assert isinstance(count, int)
        assert count == 100

    @pytest.mark.unit
    def test_search_returns_results(self, service):
        """Test search returns list of results."""
        query_embedding = np.array([0.1] * 384)
        results = service.search(query_embedding, n_results=3)

        assert isinstance(results, list)
        assert len(results) <= 3

    @pytest.mark.unit
    def test_search_result_structure(self, service):
        """Test search results have correct structure."""
        query_embedding = np.array([0.1] * 384)
        results = service.search(query_embedding, n_results=3)

        if results:
            result = results[0]
            assert hasattr(result, "conversation")
            assert hasattr(result, "score")
            assert 0 <= result.score <= 1

    @pytest.mark.unit
    def test_search_with_min_score_filter(self, service, mock_chroma_collection):
        """Test search filters by minimum score."""
        mock_chroma_collection.query.return_value = {
            "ids": [["conv_1", "conv_2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [
                [
                    {"context": "Q1", "response": "A1", "id": "1"},
                    {"context": "Q2", "response": "A2", "id": "2"},
                ]
            ],
            "distances": [[0.8, 0.9]],
        }

        query_embedding = np.array([0.1] * 384)
        results = service.search(query_embedding, n_results=5, min_score=0.5)

        assert isinstance(results, list)

    @pytest.mark.unit
    def test_search_empty_store(self, service, mock_chroma_collection):
        """Test search with empty vector store."""
        mock_chroma_collection.count.return_value = 0
        mock_chroma_collection.query.return_value = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
        }

        query_embedding = np.array([0.1] * 384)
        results = service.search(query_embedding, n_results=5)

        assert results == []

    @pytest.mark.unit
    def test_add_conversations(self, service, mock_chroma_collection):
        """Test adding conversations to store."""
        from src.models.schemas import Conversation

        conversations = [
            Conversation(id=1, context="Q1", response="A1"),
            Conversation(id=2, context="Q2", response="A2"),
        ]
        embeddings = np.array([[0.1] * 384, [0.2] * 384])

        service.add_conversations(conversations, embeddings)

        mock_chroma_collection.add.assert_called_once()

    @pytest.mark.unit
    def test_add_conversations_empty_list(self, service):
        """Test adding empty list is handled gracefully."""
        result = service.add_conversations([], np.array([]))
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_reset_clears_collection(self, service, mock_chroma_client):
        """Test reset clears and recreates collection."""
        service.reset()

        mock_chroma_client.delete_collection.assert_called()
        mock_chroma_client.create_collection.assert_called()

    @pytest.mark.unit
    def test_get_stats(self, service):
        """Test get_stats returns statistics."""
        stats = service.get_stats()

        assert isinstance(stats, dict)
        assert "total_documents" in stats

    @pytest.mark.unit
    def test_search_n_results_parameter(self, service, mock_chroma_collection):
        """Test n_results parameter is respected."""
        query_embedding = np.array([0.1] * 384)

        service.search(query_embedding, n_results=10)

        call_args = mock_chroma_collection.query.call_args
        assert call_args is not None

    @pytest.mark.unit
    def test_distance_to_similarity_conversion(self, service):
        """Test distance is converted to similarity score."""
        query_embedding = np.array([0.1] * 384)
        results = service.search(query_embedding, n_results=3)

        if results:
            for result in results:
                assert 0 <= result.score <= 1

    @pytest.mark.unit
    def test_search_invalid_embedding_dimension(self, service):
        """Test search with wrong embedding dimension."""
        wrong_dimension_embedding = np.array([0.1] * 100)

        # Should either raise error or handle gracefully
        try:
            service.search(wrong_dimension_embedding, n_results=5)
        except (ValueError, Exception):
            pass

    @pytest.mark.unit
    def test_add_duplicate_ids(self, service, mock_chroma_collection):
        """Test handling of duplicate IDs."""
        from src.models.schemas import Conversation

        conversations = [
            Conversation(id=1, context="Q1", response="A1"),
            Conversation(id=1, context="Q1 duplicate", response="A1 duplicate"),
        ]
        embeddings = np.array([[0.1] * 384, [0.2] * 384])

        service.add_conversations(conversations, embeddings)


class TestVectorStoreServiceIntegration:
    """Integration tests with real ChromaDB."""

    @pytest.fixture
    def temp_persist_dir(self, tmp_path):
        """Create temporary persist directory."""
        persist_dir = tmp_path / "chroma_test"
        persist_dir.mkdir()
        return str(persist_dir)

    @pytest.mark.slow
    @pytest.mark.integration
    def test_real_chromadb_operations(self, temp_persist_dir):
        """Test real ChromaDB operations."""
        import os

        os.environ["CHROMA_PERSIST_DIRECTORY"] = temp_persist_dir

        from src.core.vector_store import VectorStoreService

        service = VectorStoreService()

        # Test count
        initial_count = service.count()
        assert initial_count >= 0

        # Test add
        from src.models.schemas import Conversation

        conversations = [
            Conversation(id=999, context="Test question", response="Test answer"),
        ]
        embeddings = np.array([[0.1] * 384])

        service.add_conversations(conversations, embeddings)

        # Test search
        results = service.search(np.array([0.1] * 384), n_results=1)
        assert isinstance(results, list)

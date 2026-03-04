"""
Vector Store Service - Professional Reddit RAG Chatbot
ChromaDB wrapper for vector similarity search
"""

from typing import Any

import chromadb
import numpy as np
from chromadb.config import Settings as ChromaSettings

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.models.schemas import Conversation, SearchResult


logger = get_logger(__name__)


class VectorStoreService:
    """
    Vector store service using ChromaDB

    Handles storage and retrieval of conversation embeddings
    for efficient similarity search.
    """

    def __init__(self, collection_name: str | None = None, persist_directory: str | None = None):
        """
        Initialize vector store

        Args:
            collection_name: Name of the collection
            persist_directory: Directory to persist data
        """
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIRECTORY

        logger.info(f"Initializing ChromaDB at {self.persist_directory}")

        try:
            # Initialize ChromaDB client
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection(name=self.collection_name)
                logger.info(f"✓ Loaded existing collection: {self.collection_name}")
                logger.info(f"  Documents: {self.collection.count()}")

            except Exception:
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Reddit conversations with multilingual embeddings"},
                )
                logger.info(f"✓ Created new collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e!s}")
            raise

    def add_conversations(self, conversations: list[Conversation], embeddings: np.ndarray) -> bool:
        """
        Add conversations to vector store

        Args:
            conversations: List of Conversation objects
            embeddings: Corresponding embeddings array

        Returns:
            Success status
        """
        try:
            # Prepare data for ChromaDB
            ids = [f"conv_{conv.id}" for conv in conversations]
            documents = [conv.full_text for conv in conversations]
            metadatas = [
                {"context": conv.context, "response": conv.response, "id": conv.id}
                for conv in conversations
            ]

            # Add to collection
            self.collection.add(
                ids=ids, embeddings=embeddings.tolist(), documents=documents, metadatas=metadatas
            )

            logger.info(f"✓ Added {len(conversations)} conversations to vector store")
            return True

        except Exception as e:
            logger.error(f"Failed to add conversations: {e!s}")
            return False

    def search(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5,
        min_score: float = 0.0,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """
        Search for similar conversations

        Args:
            query_embedding: Query embedding vector
            n_results: Number of results to return
            min_score: Minimum similarity score (0-1)
            filters: Optional metadata filters

        Returns:
            List of SearchResult objects
        """
        try:
            # Query ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()], n_results=n_results, where=filters
            )

            # Parse results
            search_results = []

            if not results["ids"] or not results["ids"][0]:
                logger.warning("No results found")
                return []

            for i in range(len(results["ids"][0])):
                # Calculate similarity score from distance
                distance = results["distances"][0][i]
                score = 1 / (1 + distance)  # Convert distance to similarity

                # Apply minimum score filter
                if score < min_score:
                    continue

                # Create Conversation object
                metadata = results["metadatas"][0][i]
                conversation = Conversation(
                    id=metadata["id"],
                    context=metadata["context"],
                    response=metadata["response"],
                    full_text=results["documents"][0][i],
                )

                # Create SearchResult
                search_result = SearchResult(
                    conversation=conversation, score=score, distance=distance, rank=i + 1
                )

                search_results.append(search_result)

            logger.debug(f"Found {len(search_results)} results (min_score: {min_score})")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e!s}")
            return []

    def count(self) -> int:
        """
        Get total number of documents

        Returns:
            Document count
        """
        try:
            return self.collection.count()
        except Exception as e:
            logger.error(f"Count failed: {e!s}")
            return 0

    def delete_collection(self) -> bool:
        """
        Delete the collection

        Returns:
            Success status
        """
        try:
            self.client.delete_collection(name=self.collection_name)
            logger.info(f"✓ Deleted collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e!s}")
            return False

    def reset(self) -> bool:
        """
        Reset the vector store (delete and recreate)

        Returns:
            Success status
        """
        try:
            self.delete_collection()
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "Reddit conversations with multilingual embeddings"},
            )
            logger.info(f"✓ Reset collection: {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to reset: {e!s}")
            return False

    def get_stats(self) -> dict:
        """
        Get vector store statistics

        Returns:
            Statistics dictionary
        """
        return {
            "collection_name": self.collection_name,
            "total_documents": self.count(),
            "persist_directory": self.persist_directory,
        }


# Singleton instance
_vector_store_service: VectorStoreService | None = None


def get_vector_store_service() -> VectorStoreService:
    """
    Get vector store service singleton

    Returns:
        VectorStoreService instance
    """
    global _vector_store_service
    if _vector_store_service is None:
        _vector_store_service = VectorStoreService()
    return _vector_store_service

"""
Embeddings Service - Professional Reddit RAG Chatbot
Handles text embedding generation using sentence-transformers
"""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config.logging_config import get_logger
from src.config.settings import settings


logger = get_logger(__name__)


class EmbeddingService:
    """
    Service for generating text embeddings

    Uses sentence-transformers multilingual model for cross-lingual support.
    Supports caching for performance optimization.
    """

    def __init__(self, model_name: str | None = None, device: str | None = None):
        """
        Initialize embedding service

        Args:
            model_name: Embedding model name (default from settings)
            device: Device to use (cpu/cuda, default from settings)
        """
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or settings.EMBEDDING_DEVICE

        logger.info(f"Loading embedding model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()

            logger.info(f"✓ Model loaded, embedding dimension: {self.embedding_dim}")

        except Exception as e:
            logger.error(f"Failed to load embedding model: {e!s}")
            raise

    def embed_text(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Generate embedding for single text

        Args:
            text: Text to embed
            normalize: Whether to normalize embedding

        Returns:
            Embedding vector as numpy array
        """
        try:
            embedding = self.model.encode(
                text, normalize_embeddings=normalize, show_progress_bar=False, convert_to_numpy=True
            )

            logger.debug(f"Generated embedding for text: '{text[:50]}...'")
            return embedding

        except Exception as e:
            logger.error(f"Embedding generation failed: {e!s}")
            raise

    def embed_batch(
        self,
        texts: list[str],
        batch_size: int | None = None,
        normalize: bool = True,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Generate embeddings for batch of texts

        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing
            normalize: Whether to normalize embeddings
            show_progress: Whether to show progress bar

        Returns:
            Array of embeddings
        """
        batch_size = batch_size or settings.EMBEDDING_BATCH_SIZE

        try:
            logger.info(f"Generating embeddings for {len(texts)} texts...")

            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=normalize,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
            )

            logger.info(f"✓ Generated {len(embeddings)} embeddings")
            return embeddings

        except Exception as e:
            logger.error(f"Batch embedding failed: {e!s}")
            raise

    def get_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two embeddings

        Args:
            embedding1: First embedding
            embedding2: Second embedding

        Returns:
            Similarity score (0-1)
        """
        try:
            # Cosine similarity
            similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )

            return float(similarity)

        except Exception as e:
            logger.error(f"Similarity calculation failed: {e!s}")
            return 0.0

    def get_model_info(self) -> dict:
        """
        Get embedding model information

        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dim,
            "device": self.device,
            "max_seq_length": self.model.max_seq_length,
        }


# Singleton instance with lazy initialization
_embedding_service: EmbeddingService = None


@lru_cache
def get_embedding_service() -> EmbeddingService:
    """
    Get embedding service singleton

    Returns:
        EmbeddingService instance
    """
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service

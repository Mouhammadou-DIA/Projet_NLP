"""
Reranking Module for improved search relevance.
Uses cross-encoder models to rerank search results.
"""

from loguru import logger

from src.models.schemas import SearchResult


class RerankerService:
    """
    Service for reranking search results using cross-encoder models.

    Cross-encoders are more accurate than bi-encoders for ranking
    but slower, so they're used as a second-stage ranker.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str = "cpu",
        batch_size: int = 32,
    ):
        """
        Initialize the reranker service.

        Args:
            model_name: HuggingFace model name for cross-encoder.
            device: Device to run model on (cpu/cuda).
            batch_size: Batch size for inference.
        """
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = None

        self._load_model()

    def _load_model(self) -> None:
        """Load the cross-encoder model."""
        try:
            from sentence_transformers import CrossEncoder

            self.model = CrossEncoder(
                self.model_name,
                device=self.device,
            )
            logger.info(f"Reranker model loaded: {self.model_name}")
        except ImportError:
            logger.warning(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            self.model = None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        top_k: int | None = None,
    ) -> list[SearchResult]:
        """
        Rerank search results based on query relevance.

        Args:
            query: The search query.
            results: List of search results to rerank.
            top_k: Number of top results to return (None for all).

        Returns:
            Reranked list of search results.
        """
        if not results:
            return results

        if self.model is None:
            logger.warning("Reranker model not available, returning original results")
            return results

        try:
            # Prepare query-document pairs
            pairs = [[query, result.conversation.context] for result in results]

            # Get cross-encoder scores
            scores = self.model.predict(pairs, batch_size=self.batch_size)

            # Normalize scores to 0-1 for display/consistency
            scores = [float(s) for s in scores]
            min_score = min(scores)
            max_score = max(scores)
            if max_score > min_score:
                norm_scores = [(s - min_score) / (max_score - min_score) for s in scores]
            else:
                # All scores identical -> treat as neutral relevance
                norm_scores = [0.5 for _ in scores]

            # Combine results with new scores
            scored_results = list(zip(results, norm_scores))

            # Sort by cross-encoder score (higher is better)
            scored_results.sort(key=lambda x: x[1], reverse=True)

            # Update scores and ranks
            reranked = []
            for rank, (result, score) in enumerate(scored_results, 1):
                # Create new result with updated score and rank
                reranked_result = SearchResult(
                    conversation=result.conversation,
                    score=float(score),
                    rank=rank,
                )
                reranked.append(reranked_result)

            # Apply top_k if specified
            if top_k is not None:
                reranked = reranked[:top_k]

            logger.debug(f"Reranked {len(results)} results, returning top {len(reranked)}")
            return reranked

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            return results

    def score_pair(self, query: str, document: str) -> float:
        """
        Score a single query-document pair.

        Args:
            query: The query text.
            document: The document text.

        Returns:
            Relevance score.
        """
        if self.model is None:
            return 0.0

        try:
            score = self.model.predict([[query, document]])[0]
            return float(score)
        except Exception as e:
            logger.error(f"Scoring failed: {e}")
            return 0.0

    def is_available(self) -> bool:
        """Check if reranker is available."""
        return self.model is not None


class HybridSearchReranker:
    """
    Combines dense vector search with BM25 sparse search,
    then reranks results using a cross-encoder.
    """

    def __init__(
        self,
        reranker: RerankerService | None = None,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ):
        """
        Initialize hybrid search reranker.

        Args:
            reranker: Optional reranker service for final reranking.
            dense_weight: Weight for dense (vector) search scores.
            sparse_weight: Weight for sparse (BM25) search scores.
        """
        self.reranker = reranker
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        # Normalize weights
        total = dense_weight + sparse_weight
        self.dense_weight = dense_weight / total
        self.sparse_weight = sparse_weight / total

    def combine_scores(
        self,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
    ) -> list[SearchResult]:
        """
        Combine dense and sparse search results using reciprocal rank fusion.

        Args:
            dense_results: Results from dense vector search.
            sparse_results: Results from sparse (BM25) search.

        Returns:
            Combined and ranked results.
        """
        k = 60  # RRF constant

        # Build score maps
        combined_scores: dict[str, float] = {}
        result_map: dict[str, SearchResult] = {}

        # Process dense results
        for rank, result in enumerate(dense_results, 1):
            doc_id = str(result.conversation.id)
            rrf_score = self.dense_weight / (k + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
            result_map[doc_id] = result

        # Process sparse results
        for rank, result in enumerate(sparse_results, 1):
            doc_id = str(result.conversation.id)
            rrf_score = self.sparse_weight / (k + rank)
            combined_scores[doc_id] = combined_scores.get(doc_id, 0) + rrf_score
            if doc_id not in result_map:
                result_map[doc_id] = result

        # Sort by combined score
        sorted_ids = sorted(
            combined_scores.keys(),
            key=lambda x: combined_scores[x],
            reverse=True,
        )

        # Build final results
        combined_results = []
        for rank, doc_id in enumerate(sorted_ids, 1):
            result = result_map[doc_id]
            combined_result = SearchResult(
                conversation=result.conversation,
                score=combined_scores[doc_id],
                rank=rank,
            )
            combined_results.append(combined_result)

        return combined_results

    def search_and_rerank(
        self,
        query: str,
        dense_results: list[SearchResult],
        sparse_results: list[SearchResult],
        top_k: int = 10,
        rerank_top_n: int = 50,
    ) -> list[SearchResult]:
        """
        Combine search results and optionally rerank.

        Args:
            query: The search query.
            dense_results: Results from dense search.
            sparse_results: Results from sparse search.
            top_k: Final number of results to return.
            rerank_top_n: Number of results to pass to reranker.

        Returns:
            Final ranked results.
        """
        # Combine results
        combined = self.combine_scores(dense_results, sparse_results)

        # Limit to rerank_top_n before expensive reranking
        candidates = combined[:rerank_top_n]

        # Rerank if available
        if self.reranker and self.reranker.is_available():
            return self.reranker.rerank(query, candidates, top_k=top_k)

        return candidates[:top_k]


# Singleton instance
_reranker: RerankerService | None = None


def get_reranker(
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    device: str = "cpu",
) -> RerankerService:
    """
    Get or create the global reranker instance.

    Args:
        model_name: Model name for cross-encoder.
        device: Device to run on.

    Returns:
        RerankerService instance.
    """
    global _reranker
    if _reranker is None:
        _reranker = RerankerService(model_name=model_name, device=device)
    return _reranker

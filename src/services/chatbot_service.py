"""
Chatbot Service - Professional Reddit RAG Chatbot
Main business logic with reranking, caching, conversation memory, and monitoring
"""

import time
from datetime import datetime

from src.config.logging_config import get_logger, log_metric
from src.config.settings import settings
from src.core.cache import CacheService, get_cache_service, make_cache_key
from src.core.conversation_memory import (
    ConversationMemory,
    SummarizingMemory,
    get_conversation_memory,
)
from src.core.embeddings import EmbeddingService
from src.core.llm_handler import LLMService
from src.core.reranker import RerankerService, get_reranker
from src.core.vector_store import VectorStoreService
from src.models.schemas import ChatRequest, ChatResponse, SearchResult
from src.utils.text_processor import TextProcessor
from src.utils.validators import validate_input


logger = get_logger(__name__)


class ChatbotService:
    """
    Professional Chatbot Service

    Handles RAG workflow:
    1. Embed user query
    2. Search similar conversations
    3. Rerank results with cross-encoder
    4. Cache results for repeated queries
    5. Manage conversation memory across sessions
    6. Generate response (simple or LLM)
    7. Return response with metadata
    """

    def __init__(
        self,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStoreService | None = None,
        llm_service: LLMService | None = None,
        reranker: RerankerService | None = None,
        cache_service: CacheService | None = None,
        conversation_memory: ConversationMemory | None = None,
    ):
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_store = vector_store or VectorStoreService()
        self.llm_service = llm_service or LLMService()
        self.text_processor = TextProcessor()

        # Reranker (cross-encoder for improved relevance)
        if reranker is not None:
            self.reranker = reranker
        elif settings.RERANKER_ENABLED:
            try:
                self.reranker = get_reranker(
                    model_name=settings.RERANKER_MODEL,
                    device=settings.RERANKER_DEVICE,
                )
            except Exception as e:
                logger.warning(f"Failed to initialize reranker: {e}")
                self.reranker = None
        else:
            self.reranker = None

        # Cache service
        self.cache = cache_service or get_cache_service(
            backend_type=settings.CACHE_BACKEND,
            redis_url=settings.REDIS_URL,
            enabled=settings.ENABLE_CACHING,
        )

        # Conversation memory with summarization
        base_memory = conversation_memory or get_conversation_memory(
            max_sessions=settings.MEMORY_MAX_SESSIONS,
            session_timeout=settings.MEMORY_SESSION_TIMEOUT,
        )
        self.memory = base_memory

        # Summarizing memory wraps base memory to condense old messages
        self.summarizing_memory = SummarizingMemory(
            base_memory=base_memory,
            summarizer=self._summarize_with_llm,
            summary_threshold=settings.MEMORY_SUMMARY_THRESHOLD,
            keep_recent=settings.MEMORY_KEEP_RECENT,
        )

        logger.info(
            f"ChatbotService initialized "
            f"(reranker={'enabled' if self.reranker and self.reranker.is_available() else 'disabled'}, "
            f"cache={'enabled' if self.cache.enabled else 'disabled'}, "
            f"memory=enabled)"
        )

    def chat(self, request: ChatRequest, session_id: str | None = None) -> ChatResponse:
        """
        Main chat function with reranking, caching, and conversation memory.

        Args:
            request: Chat request with user message and parameters
            session_id: Optional session ID for conversation continuity

        Returns:
            ChatResponse with message, sources, and metadata
        """
        start_time = time.time()

        try:
            # 1. Validate input
            validate_input(request.message, max_length=1000)
            logger.info(f"Processing chat request: '{request.message[:50]}...'")

            # 2. Manage conversation session
            session = self.memory.get_or_create_session(session_id)
            session_id = session.session_id

            # Store user message in memory
            self.memory.add_message(session_id, "user", request.message)

            # 3. Check cache for identical query
            cache_key = make_cache_key(
                request.message,
                use_llm=request.use_llm,
                n_results=request.n_results,
            )
            cached_response = self.cache.get(cache_key)
            if cached_response is not None:
                logger.info("Cache hit - returning cached response")
                duration = (time.time() - start_time) * 1000
                cached_response["metadata"]["duration_ms"] = round(duration, 2)
                cached_response["metadata"]["cache_hit"] = True
                cached_response["metadata"]["session_id"] = session_id

                # Store assistant response in memory
                self.memory.add_message(session_id, "assistant", cached_response["message"])

                return ChatResponse(**cached_response)

            # 4. Search similar conversations
            search_results = self._search_similar(
                query=request.message, n_results=request.n_results
            )

            # 5. Rerank results with cross-encoder
            if self.reranker and self.reranker.is_available() and search_results:
                rerank_start = time.time()
                search_results = self.reranker.rerank(
                    query=request.message,
                    results=search_results,
                    top_k=settings.RERANKER_TOP_K,
                )
                rerank_duration = (time.time() - rerank_start) * 1000
                logger.info(f"Reranked results in {rerank_duration:.2f}ms")
                log_metric("rerank_duration_ms", rerank_duration)

            # 6. Build conversation context from memory
            memory_context = self.summarizing_memory.get_context(
                session_id=session_id,
                include_summary=True,
            )

            # 7. Generate response
            if request.use_llm and self.llm_service.is_available():
                # Merge conversation_history from request with memory context
                history = request.conversation_history
                if not history and memory_context:
                    # Use memory-based context if no explicit history provided
                    pass  # memory_context will be injected via _generate_with_llm

                response_text = self._generate_with_llm(
                    query=request.message,
                    context=search_results,
                    history=request.conversation_history,
                    memory_context=memory_context,
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                )
            else:
                response_text = self._generate_simple(search_results)

            # 8. Store assistant response in memory
            self.memory.add_message(session_id, "assistant", response_text)

            # 9. Build response
            duration = (time.time() - start_time) * 1000

            response = ChatResponse(
                message=response_text,
                sources=search_results[:3],
                metadata={
                    "duration_ms": round(duration, 2),
                    "method": "llm" if request.use_llm else "simple",
                    "n_sources": len(search_results),
                    "model": settings.LLM_MODEL if request.use_llm else "retrieval",
                    "reranked": bool(self.reranker and self.reranker.is_available()),
                    "cache_hit": False,
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # 10. Cache the response
            self.cache.set(cache_key, response.dict(), ttl=settings.CACHE_TTL)

            # 11. Log metrics
            log_metric(
                "chat_duration_ms", duration, {"method": "llm" if request.use_llm else "simple"}
            )
            log_metric("sources_retrieved", len(search_results))

            logger.info(f"Chat completed in {duration:.2f}ms")
            return response

        except Exception as e:
            logger.error(f"Chat failed: {e!s}")
            duration = (time.time() - start_time) * 1000
            log_metric("chat_error", 1, {"error_type": type(e).__name__})
            raise

    def _search_similar(self, query: str, n_results: int = 5) -> list[SearchResult]:
        """Search for similar conversations."""
        try:
            processed_query = self.text_processor.clean_text(query)
            query_embedding = self.embedding_service.embed_text(processed_query)

            # When reranker is enabled, fetch more candidates for better reranking
            fetch_n = n_results
            if self.reranker and self.reranker.is_available():
                fetch_n = max(n_results * 3, 15)

            results = self.vector_store.search(
                query_embedding=query_embedding,
                n_results=fetch_n,
                min_score=settings.MIN_SIMILARITY_SCORE,
            )

            logger.debug(f"Found {len(results)} similar conversations")
            return results

        except Exception as e:
            logger.error(f"Search failed: {e!s}")
            raise

    def _generate_simple(self, search_results: list[SearchResult]) -> str:
        """Generate simple response (best match)."""
        if not search_results:
            return "I couldn't find any relevant conversation. Could you rephrase your question?"

        best_match = search_results[0]
        logger.debug(f"Using best match (score: {best_match.score:.3f})")
        return best_match.conversation.response

    def _generate_with_llm(
        self,
        query: str,
        context: list[SearchResult],
        history: list | None = None,
        memory_context: str = "",
        temperature: float = 0.7,
        max_tokens: int = 500,
    ) -> str:
        """Generate response using LLM with conversation memory context."""
        try:
            context_text = self._build_context(context)

            # Inject memory context into the prompt if available
            if memory_context:
                context_text = (
                    f"Previous conversation:\n{memory_context}\n\n"
                    f"Relevant Reddit conversations:\n{context_text}"
                )

            response = self.llm_service.generate(
                query=query,
                context=context_text,
                history=history or [],
                temperature=temperature,
                max_tokens=max_tokens,
            )

            logger.debug(f"Generated LLM response ({len(response)} chars)")
            return response

        except Exception as e:
            logger.warning(f"LLM generation failed: {e!s}, falling back to simple")
            return self._generate_simple(context)

    def _build_context(self, search_results: list[SearchResult]) -> str:
        """Build context string from search results."""
        context_parts = []

        for i, result in enumerate(search_results[:3], 1):
            conv = result.conversation
            if 0.0 <= result.score <= 1.0:
                score_label = f"{result.score:.0%}"
            else:
                score_label = f"{result.score:.3f}"
            context_parts.append(
                f"Example {i} (relevance: {score_label}):\nQ: {conv.context}\nA: {conv.response}"
            )

        return "\n\n".join(context_parts)

    def _summarize_with_llm(self, text: str) -> str | None:
        """
        Summarize conversation history using the LLM.
        Used by SummarizingMemory when the history exceeds the threshold.
        """
        if not self.llm_service.is_available():
            return None

        try:
            summary = self.llm_service.generate(
                query="Summarize the following conversation in 2-3 concise sentences:",
                context=text,
                history=[],
                temperature=0.3,
                max_tokens=150,
            )
            return summary
        except Exception as e:
            logger.warning(f"Failed to summarize conversation: {e}")
            return None

    def get_stats(self) -> dict:
        """Get chatbot statistics including cache and memory info."""
        try:
            stats = {
                "total_conversations": self.vector_store.count(),
                "embedding_model": settings.EMBEDDING_MODEL,
                "llm_model": settings.LLM_MODEL,
                "llm_available": self.llm_service.is_available(),
                "reranker_enabled": bool(self.reranker and self.reranker.is_available()),
                "reranker_model": settings.RERANKER_MODEL if self.reranker else None,
                "cache_enabled": self.cache.enabled,
                "cache_stats": self.cache.get_stats(),
                "memory_stats": self.memory.get_stats(),
            }

            logger.debug(f"Stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get stats: {e!s}")
            return {}

    def health_check(self) -> dict:
        """Check service health."""
        health = {
            "embedding_service": "healthy",
            "vector_store": "healthy",
            "llm_service": "healthy" if self.llm_service.is_available() else "unavailable",
            "reranker": "healthy"
            if self.reranker and self.reranker.is_available()
            else "unavailable",
            "cache": "healthy" if self.cache.enabled else "disabled",
        }

        try:
            self.embedding_service.embed_text("test")
        except Exception:
            health["embedding_service"] = "unhealthy"

        try:
            self.vector_store.count()
        except Exception:
            health["vector_store"] = "unhealthy"

        return health


# Singleton instance with lazy initialization
_chatbot_service: ChatbotService | None = None


def get_chatbot_service() -> ChatbotService:
    """Get chatbot service singleton."""
    global _chatbot_service
    if _chatbot_service is None:
        _chatbot_service = ChatbotService()
    return _chatbot_service

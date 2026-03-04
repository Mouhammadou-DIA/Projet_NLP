"""
LLM Handler Service - Professional Reddit RAG Chatbot
Handles LLM integration (Ollama, OpenAI, Anthropic)
"""

from enum import Enum

from src.config.logging_config import get_logger
from src.config.settings import settings
from src.models.schemas import ChatMessage


logger = get_logger(__name__)


class LLMProvider(str, Enum):
    """LLM provider enum"""

    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"


class LLMService:
    """
    LLM service with multiple provider support

    Supports:
    - Ollama (local, free)
    - OpenAI (API)
    - Anthropic (API)
    """

    def __init__(self, provider: str | None = None, model: str | None = None):
        """
        Initialize LLM service

        Args:
            provider: LLM provider (ollama/openai/anthropic)
            model: Model name
        """
        self.provider = provider or settings.LLM_PROVIDER
        self.model = model or settings.LLM_MODEL

        logger.info(f"Initializing LLM service: {self.provider}/{self.model}")

        self._available = self._check_availability()

        if self._available:
            logger.info(f"✓ LLM service ready: {self.provider}")
        else:
            logger.warning(f"⚠ LLM service not available: {self.provider}")

    def _check_availability(self) -> bool:
        """
        Check if LLM provider is available

        Returns:
            Availability status
        """
        try:
            if self.provider == LLMProvider.OLLAMA:
                import ollama

                # Try to list models to check if Ollama is running
                ollama.list()
                return True

            elif self.provider == LLMProvider.OPENAI:
                # Check if API key is set
                import os

                return bool(os.getenv("OPENAI_API_KEY"))

            elif self.provider == LLMProvider.ANTHROPIC:
                import os

                return bool(os.getenv("ANTHROPIC_API_KEY"))

            elif self.provider == LLMProvider.GROQ:
                # Check via settings (loaded from .env)
                return bool(settings.GROQ_API_KEY)

            return False

        except Exception as e:
            logger.debug(f"LLM availability check failed: {e!s}")
            return False

    def is_available(self) -> bool:
        """Check if LLM is available"""
        return self._available

    def generate(
        self,
        query: str,
        context: str,
        history: list[ChatMessage] | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """
        Generate response using LLM

        Args:
            query: User query
            context: Context from retrieved conversations
            history: Conversation history
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate

        Returns:
            Generated response
        """
        temperature = temperature or settings.LLM_TEMPERATURE
        max_tokens = max_tokens or settings.LLM_MAX_TOKENS

        if not self._available:
            raise RuntimeError(f"LLM provider {self.provider} not available")

        try:
            if self.provider == LLMProvider.OLLAMA:
                return self._generate_ollama(query, context, history, temperature, max_tokens)
            elif self.provider == LLMProvider.OPENAI:
                return self._generate_openai(query, context, history, temperature, max_tokens)
            elif self.provider == LLMProvider.ANTHROPIC:
                return self._generate_anthropic(query, context, history, temperature, max_tokens)
            elif self.provider == LLMProvider.GROQ:
                return self._generate_groq(query, context, history, temperature, max_tokens)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")

        except Exception as e:
            logger.error(f"LLM generation failed: {e!s}")
            raise

    def _generate_ollama(
        self,
        query: str,
        context: str,
        history: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate with Ollama (Meta Llama 3.1)"""
        import ollama

        # Build messages for chat format (better for Llama 3.1)
        messages = [{"role": "system", "content": self._get_system_prompt()}]

        # Add conversation history if available
        if history:
            for msg in history[-5:]:  # Last 5 messages
                messages.append({"role": msg.role, "content": msg.content})

        # Add context and current query
        user_content = self._build_user_message(query, context)
        messages.append({"role": "user", "content": user_content})

        # Call Ollama with Llama 3.1
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
            },
        )

        return response["message"]["content"]

    def _build_user_message(self, query: str, context: str) -> str:
        """Build user message with context for Llama 3.1"""
        if context:
            return (
                f"Context from Reddit conversations (in English):\n\n"
                f"{context}\n\n"
                f"User question: {query}\n\n"
                f"IMPORTANT: Respond in the SAME LANGUAGE as the user's question above. "
                f"If the question is in French, your entire response must be in French."
            )
        return query

    def _generate_openai(
        self,
        query: str,
        context: str,
        history: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate with OpenAI"""
        from openai import OpenAI

        client = OpenAI()

        # Build messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._build_prompt(query, context, history)},
        ]

        # Call OpenAI
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        query: str,
        context: str,
        history: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate with Anthropic"""
        import anthropic

        client = anthropic.Anthropic()

        # Build prompt
        prompt = self._build_prompt(query, context, history)

        # Call Anthropic
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )

        return response.content[0].text

    def _generate_groq(
        self,
        query: str,
        context: str,
        history: list[ChatMessage],
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate with Groq (FREE and FAST!)"""

        from groq import Groq

        client = Groq(api_key=settings.GROQ_API_KEY)

        # Build messages
        messages = [
            {"role": "system", "content": self._get_system_prompt()},
            {"role": "user", "content": self._build_user_message(query, context)},
        ]

        # Call Groq
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    def _build_prompt(
        self, query: str, context: str, history: list[ChatMessage] | None = None
    ) -> str:
        """
        Build prompt for LLM

        Args:
            query: User query
            context: Retrieved context
            history: Conversation history

        Returns:
            Formatted prompt
        """
        prompt_parts = []

        # System instructions
        prompt_parts.append(
            "You are a helpful conversational AI assistant based on Reddit conversations. "
            "Use the following examples to provide natural, helpful responses. "
            "If the question is in French, you can respond in French. "
            "Keep responses concise and conversational."
        )

        # Add context
        if context:
            prompt_parts.append(f"\n\nRelevant examples from Reddit:\n\n{context}")

        # Add history if available
        if history:
            prompt_parts.append("\n\nConversation history:")
            for msg in history[-5:]:  # Last 5 messages
                prompt_parts.append(f"{msg.role}: {msg.content}")

        # Add current query
        prompt_parts.append(f"\n\nUser question: {query}")
        prompt_parts.append("\nYour response:")

        return "\n".join(prompt_parts)

    def _get_system_prompt(self) -> str:
        """Get system prompt for chat models (optimized for Llama 3.1)"""
        return (
            "You are a friendly and helpful conversational AI assistant. "
            "Your responses are based on real Reddit conversations.\n\n"
            "CRITICAL RULES:\n"
            "1. LANGUAGE: Respond in the SAME language as the user's question. "
            "If they write in French, respond in French. If in English, respond in English.\n"
            "2. DO NOT add labels like 'French!', 'Translation:', or any meta-commentary.\n"
            "3. DO NOT translate your response - just give ONE answer in the user's language.\n"
            "4. Be concise, natural and conversational.\n"
            "5. Use the provided context to give relevant answers."
        )


# Singleton instance
_llm_service: LLMService | None = None


def get_llm_service() -> LLMService:
    """
    Get LLM service singleton

    Returns:
        LLMService instance
    """
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

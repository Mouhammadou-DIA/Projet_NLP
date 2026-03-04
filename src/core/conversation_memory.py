"""
Conversation Memory Module for multi-turn conversations.
Maintains context across multiple interactions.
"""

import contextlib
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from loguru import logger


@dataclass
class Message:
    """A single message in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for a conversation session."""

    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def add_message(self, role: str, content: str, **metadata) -> Message:
        """Add a message to the conversation."""
        message = Message(
            role=role,
            content=content,
            metadata=metadata,
        )
        self.messages.append(message)
        self.last_activity = time.time()
        return message

    def get_history(self, max_messages: int | None = None) -> list[Message]:
        """Get conversation history."""
        if max_messages:
            return self.messages[-max_messages:]
        return self.messages

    def get_context_string(
        self,
        max_messages: int = 10,
        max_chars: int = 2000,
    ) -> str:
        """
        Get conversation context as a formatted string.

        Args:
            max_messages: Maximum number of recent messages.
            max_chars: Maximum total characters.

        Returns:
            Formatted conversation history.
        """
        history = self.get_history(max_messages)

        lines = []
        total_chars = 0

        for msg in reversed(history):
            line = f"{msg.role.capitalize()}: {msg.content}"
            if total_chars + len(line) > max_chars:
                break
            lines.insert(0, line)
            total_chars += len(line) + 1

        return "\n".join(lines)

    def clear(self) -> None:
        """Clear conversation history."""
        self.messages.clear()
        self.last_activity = time.time()

    @property
    def message_count(self) -> int:
        """Get number of messages."""
        return len(self.messages)

    @property
    def is_empty(self) -> bool:
        """Check if conversation is empty."""
        return len(self.messages) == 0


class ConversationMemory:
    """
    Manages multiple conversation sessions.
    Provides session management and context retrieval.
    """

    def __init__(
        self,
        max_sessions: int = 1000,
        session_timeout: int = 3600,  # 1 hour
        max_messages_per_session: int = 100,
    ):
        """
        Initialize conversation memory.

        Args:
            max_sessions: Maximum number of concurrent sessions.
            session_timeout: Session timeout in seconds.
            max_messages_per_session: Max messages per session.
        """
        self.max_sessions = max_sessions
        self.session_timeout = session_timeout
        self.max_messages_per_session = max_messages_per_session

        self._sessions: dict[str, ConversationContext] = {}
        self._session_order: deque = deque()

    def create_session(self, session_id: str | None = None) -> str:
        """
        Create a new conversation session.

        Args:
            session_id: Optional session ID (generated if not provided).

        Returns:
            Session ID.
        """
        if session_id is None:
            session_id = str(uuid4())

        # Evict old sessions if at capacity
        while len(self._sessions) >= self.max_sessions:
            self._evict_oldest()

        context = ConversationContext(session_id=session_id)
        self._sessions[session_id] = context
        self._session_order.append(session_id)

        logger.debug(f"Created conversation session: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> ConversationContext | None:
        """
        Get a conversation session.

        Args:
            session_id: Session ID.

        Returns:
            ConversationContext or None if not found/expired.
        """
        context = self._sessions.get(session_id)

        if context is None:
            return None

        # Check if session has expired
        if time.time() - context.last_activity > self.session_timeout:
            self.delete_session(session_id)
            return None

        return context

    def get_or_create_session(self, session_id: str | None = None) -> ConversationContext:
        """
        Get existing session or create new one.

        Args:
            session_id: Optional session ID.

        Returns:
            ConversationContext.
        """
        if session_id:
            context = self.get_session(session_id)
            if context:
                return context

        new_id = self.create_session(session_id)
        return self._sessions[new_id]

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        **metadata,
    ) -> Message | None:
        """
        Add a message to a session.

        Args:
            session_id: Session ID.
            role: Message role ("user" or "assistant").
            content: Message content.
            **metadata: Additional metadata.

        Returns:
            Added Message or None if session not found.
        """
        context = self.get_session(session_id)
        if context is None:
            return None

        # Trim old messages if at limit
        while context.message_count >= self.max_messages_per_session:
            context.messages.pop(0)

        return context.add_message(role, content, **metadata)

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a conversation session.

        Args:
            session_id: Session ID.

        Returns:
            True if deleted, False if not found.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            with contextlib.suppress(ValueError):
                self._session_order.remove(session_id)
            logger.debug(f"Deleted conversation session: {session_id}")
            return True
        return False

    def _evict_oldest(self) -> None:
        """Evict the oldest session."""
        if self._session_order:
            oldest_id = self._session_order.popleft()
            self._sessions.pop(oldest_id, None)
            logger.debug(f"Evicted old session: {oldest_id}")

    def cleanup_expired(self) -> int:
        """
        Clean up expired sessions.

        Returns:
            Number of sessions cleaned up.
        """
        now = time.time()
        expired = []

        for session_id, context in self._sessions.items():
            if now - context.last_activity > self.session_timeout:
                expired.append(session_id)

        for session_id in expired:
            self.delete_session(session_id)

        if expired:
            logger.info(f"Cleaned up {len(expired)} expired sessions")

        return len(expired)

    def get_stats(self) -> dict:
        """Get memory statistics."""
        total_messages = sum(ctx.message_count for ctx in self._sessions.values())
        return {
            "active_sessions": len(self._sessions),
            "max_sessions": self.max_sessions,
            "total_messages": total_messages,
            "session_timeout": self.session_timeout,
        }


class SummarizingMemory:
    """
    Memory that summarizes old messages to maintain context
    while staying within token limits.
    """

    def __init__(
        self,
        base_memory: ConversationMemory,
        summarizer: Callable | None = None,
        summary_threshold: int = 10,
        keep_recent: int = 5,
    ):
        """
        Initialize summarizing memory.

        Args:
            base_memory: Underlying conversation memory.
            summarizer: Function to summarize messages.
            summary_threshold: Number of messages before summarizing.
            keep_recent: Number of recent messages to always keep.
        """
        self.base_memory = base_memory
        self.summarizer = summarizer
        self.summary_threshold = summary_threshold
        self.keep_recent = keep_recent

        self._summaries: dict[str, str] = {}

    def get_context(
        self,
        session_id: str,
        include_summary: bool = True,
    ) -> str:
        """
        Get conversation context with optional summary.

        Args:
            session_id: Session ID.
            include_summary: Whether to include summary of old messages.

        Returns:
            Formatted context string.
        """
        context = self.base_memory.get_session(session_id)
        if context is None:
            return ""

        # Check if we need to summarize
        if context.message_count > self.summary_threshold and self.summarizer:
            self._maybe_summarize(session_id, context)

        parts = []

        # Add summary if available
        if include_summary and session_id in self._summaries:
            parts.append(f"[Previous conversation summary: {self._summaries[session_id]}]")

        # Add recent messages
        recent = context.get_history(self.keep_recent)
        for msg in recent:
            parts.append(f"{msg.role.capitalize()}: {msg.content}")

        return "\n".join(parts)

    def _maybe_summarize(
        self,
        session_id: str,
        context: ConversationContext,
    ) -> None:
        """Summarize old messages if threshold reached."""
        if not self.summarizer:
            return

        # Get messages to summarize (all except recent)
        messages_to_summarize = context.messages[: -self.keep_recent]

        if not messages_to_summarize:
            return

        # Create text to summarize
        text = "\n".join(f"{m.role}: {m.content}" for m in messages_to_summarize)

        try:
            summary = self.summarizer(text)
            self._summaries[session_id] = summary

            # Remove summarized messages
            context.messages = context.messages[-self.keep_recent :]

            logger.debug(
                f"Summarized {len(messages_to_summarize)} messages for session {session_id}"
            )
        except Exception as e:
            logger.error(f"Failed to summarize messages: {e}")


# Global conversation memory instance
_conversation_memory: ConversationMemory | None = None


def get_conversation_memory(
    max_sessions: int = 1000,
    session_timeout: int = 3600,
) -> ConversationMemory:
    """
    Get or create the global conversation memory.

    Args:
        max_sessions: Maximum concurrent sessions.
        session_timeout: Session timeout in seconds.

    Returns:
        ConversationMemory instance.
    """
    global _conversation_memory

    if _conversation_memory is None:
        _conversation_memory = ConversationMemory(
            max_sessions=max_sessions,
            session_timeout=session_timeout,
        )

    return _conversation_memory

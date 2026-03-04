"""
Validators - Professional Reddit RAG Chatbot
Input validation utilities
"""

import re

from src.config.logging_config import get_logger


logger = get_logger(__name__)


def validate_input(
    text: str, min_length: int = 1, max_length: int = 1000, allow_empty: bool = False
) -> bool:
    """
    Validate user input text

    Args:
        text: Text to validate
        min_length: Minimum length
        max_length: Maximum length
        allow_empty: Whether to allow empty input

    Returns:
        Validation result

    Raises:
        ValueError: If validation fails
    """
    if not text:
        if allow_empty:
            return True
        raise ValueError("Input cannot be empty")

    if not isinstance(text, str):
        raise ValueError("Input must be a string")

    text_length = len(text.strip())

    if text_length < min_length:
        raise ValueError(f"Input too short (minimum {min_length} characters)")

    if text_length > max_length:
        raise ValueError(f"Input too long (maximum {max_length} characters)")

    return True


def validate_n_results(n: int, max_results: int = 20) -> bool:
    """
    Validate number of results parameter

    Args:
        n: Number of results
        max_results: Maximum allowed results

    Returns:
        Validation result

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(n, int):
        raise ValueError("n_results must be an integer")

    if n < 1:
        raise ValueError("n_results must be at least 1")

    if n > max_results:
        raise ValueError(f"n_results cannot exceed {max_results}")

    return True


def validate_temperature(temp: float) -> bool:
    """
    Validate LLM temperature parameter

    Args:
        temp: Temperature value

    Returns:
        Validation result

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(temp, (int, float)):
        raise ValueError("Temperature must be a number")

    if temp < 0 or temp > 2:
        raise ValueError("Temperature must be between 0 and 2")

    return True


def validate_max_tokens(tokens: int) -> bool:
    """
    Validate max tokens parameter

    Args:
        tokens: Max tokens value

    Returns:
        Validation result

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(tokens, int):
        raise ValueError("max_tokens must be an integer")

    if tokens < 1:
        raise ValueError("max_tokens must be at least 1")

    if tokens > 4000:
        raise ValueError("max_tokens cannot exceed 4000")

    return True


def sanitize_input(text: str) -> str:
    """
    Sanitize user input

    Args:
        text: Text to sanitize

    Returns:
        Sanitized text
    """
    if not text:
        return ""

    # Remove control characters
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)

    # Trim whitespace
    text = text.strip()

    return text


def is_potentially_harmful(text: str) -> bool:
    """
    Check if input contains potentially harmful content

    Args:
        text: Text to check

    Returns:
        True if potentially harmful
    """
    text_lower = text.lower()

    # Check for SQL injection patterns
    sql_patterns = ["drop table", "delete from", "insert into", "update set", "--", ";--"]
    if any(pattern in text_lower for pattern in sql_patterns):
        logger.warning("Potentially harmful SQL pattern detected in input")
        return True

    # Check for script injection
    script_patterns = ["<script", "javascript:", "onerror=", "onclick="]
    if any(pattern in text_lower for pattern in script_patterns):
        logger.warning("Potentially harmful script pattern detected in input")
        return True

    return False


def validate_conversation_history(history: list, max_messages: int = 50) -> bool:
    """
    Validate conversation history

    Args:
        history: List of messages
        max_messages: Maximum number of messages

    Returns:
        Validation result

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(history, list):
        raise ValueError("History must be a list")

    if len(history) > max_messages:
        raise ValueError(f"History too long (maximum {max_messages} messages)")

    return True

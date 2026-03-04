"""
Text Processor - Professional Reddit RAG Chatbot
Text cleaning and preprocessing utilities
"""

import re

from src.config.logging_config import get_logger


logger = get_logger(__name__)


class TextProcessor:
    """
    Text processing utilities

    Handles:
    - Text cleaning
    - Normalization
    - HTML/Markdown removal
    - Whitespace handling
    """

    def __init__(self):
        """Initialize text processor"""
        logger.debug("TextProcessor initialized")

    def clean_text(self, text: str, aggressive: bool = False) -> str:
        """
        Clean and normalize text

        Args:
            text: Text to clean
            aggressive: If True, apply aggressive cleaning

        Returns:
            Cleaned text
        """
        if not text:
            return ""

        # Convert to string
        text = str(text)

        # Remove HTML entities
        text = self._remove_html_entities(text)

        # Remove excessive whitespace
        text = self._normalize_whitespace(text)

        if aggressive:
            # Remove URLs
            text = self._remove_urls(text)

            # Remove special characters (keep basic punctuation)
            text = re.sub(r"[^\w\s.,!?;:\-\'\"()\[\]{}]", "", text)

        # Trim
        text = text.strip()

        return text

    def _remove_html_entities(self, text: str) -> str:
        """Remove HTML entities"""
        replacements = {
            "&gt;": ">",
            "&lt;": "<",
            "&amp;": "&",
            "&quot;": '"',
            "&apos;": "'",
            "&nbsp;": " ",
            "&#39;": "'",
        }

        for entity, replacement in replacements.items():
            text = text.replace(entity, replacement)

        return text

    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple spaces with single space
        text = re.sub(r"\s+", " ", text)

        # Remove leading/trailing whitespace
        text = text.strip()

        return text

    def _remove_urls(self, text: str) -> str:
        """Remove URLs from text"""
        # Remove http(s) URLs
        text = re.sub(r"http[s]?://\S+", "", text)

        # Remove www URLs
        text = re.sub(r"www\.\S+", "", text)

        return text

    def truncate(self, text: str, max_length: int, suffix: str = "...") -> str:
        """
        Truncate text to maximum length

        Args:
            text: Text to truncate
            max_length: Maximum length
            suffix: Suffix to add if truncated

        Returns:
            Truncated text
        """
        if len(text) <= max_length:
            return text

        return text[: max_length - len(suffix)] + suffix

    def is_valid_text(self, text: str, min_length: int = 1, max_length: int = 10000) -> bool:
        """
        Check if text is valid

        Args:
            text: Text to validate
            min_length: Minimum length
            max_length: Maximum length

        Returns:
            Validation result
        """
        if not text or not isinstance(text, str):
            return False

        text = text.strip()

        return not (len(text) < min_length or len(text) > max_length)

    def extract_keywords(self, text: str, max_keywords: int = 10) -> list:
        """
        Extract simple keywords from text

        Args:
            text: Text to extract keywords from
            max_keywords: Maximum number of keywords

        Returns:
            List of keywords
        """
        # Simple keyword extraction (lowercase, remove punctuation)
        words = re.findall(r"\b\w+\b", text.lower())

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "is",
            "was",
            "are",
            "were",
            "be",
            "been",
        }

        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        # Get unique keywords, preserve order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords[:max_keywords]

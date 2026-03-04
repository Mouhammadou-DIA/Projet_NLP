"""
Rate Limiting Module for API Protection.
Implements token bucket algorithm for request rate limiting.
"""

import time
from collections import defaultdict
from dataclasses import dataclass, field

from loguru import logger


@dataclass
class TokenBucket:
    """Token bucket for rate limiting."""

    capacity: int
    refill_rate: float  # tokens per second
    tokens: float = field(default=0)
    last_refill: float = field(default_factory=time.time)

    def __post_init__(self):
        self.tokens = float(self.capacity)

    def consume(self, tokens: int = 1) -> bool:
        """
        Try to consume tokens from the bucket.

        Args:
            tokens: Number of tokens to consume.

        Returns:
            True if tokens were consumed, False if not enough tokens.
        """
        self._refill()

        if self.tokens >= tokens:
            self.tokens -= tokens
            return True
        return False

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_refill
        self.tokens = min(self.capacity, self.tokens + elapsed * self.refill_rate)
        self.last_refill = now

    @property
    def available_tokens(self) -> int:
        """Get current available tokens."""
        self._refill()
        return int(self.tokens)


class RateLimiter:
    """
    Rate limiter using token bucket algorithm.

    Supports per-client rate limiting based on IP address or API key.
    """

    def __init__(
        self,
        requests_per_minute: int = 60,
        burst_size: int | None = None,
        enabled: bool = True,
    ):
        """
        Initialize rate limiter.

        Args:
            requests_per_minute: Maximum requests allowed per minute.
            burst_size: Maximum burst size (defaults to requests_per_minute).
            enabled: Whether rate limiting is enabled.
        """
        self.requests_per_minute = requests_per_minute
        self.burst_size = burst_size or requests_per_minute
        self.enabled = enabled
        self.refill_rate = requests_per_minute / 60.0  # tokens per second

        self._buckets: dict[str, TokenBucket] = defaultdict(self._create_bucket)
        self._blocked_until: dict[str, float] = {}

        logger.info(
            f"Rate limiter initialized: {requests_per_minute} req/min, "
            f"burst: {self.burst_size}, enabled: {enabled}"
        )

    def _create_bucket(self) -> TokenBucket:
        """Create a new token bucket."""
        return TokenBucket(
            capacity=self.burst_size,
            refill_rate=self.refill_rate,
        )

    def is_allowed(self, client_id: str) -> bool:
        """
        Check if a request from the client is allowed.

        Args:
            client_id: Client identifier (IP address or API key).

        Returns:
            True if request is allowed, False if rate limited.
        """
        if not self.enabled:
            return True

        # Check if client is temporarily blocked
        if client_id in self._blocked_until:
            if time.time() < self._blocked_until[client_id]:
                return False
            del self._blocked_until[client_id]

        # Try to consume a token
        bucket = self._buckets[client_id]
        allowed = bucket.consume(1)

        if not allowed:
            logger.warning(f"Rate limit exceeded for client: {client_id}")

        return allowed

    def get_remaining(self, client_id: str) -> int:
        """
        Get remaining requests for a client.

        Args:
            client_id: Client identifier.

        Returns:
            Number of remaining requests.
        """
        if not self.enabled:
            return self.burst_size

        bucket = self._buckets[client_id]
        return bucket.available_tokens

    def get_reset_time(self, client_id: str) -> float:
        """
        Get time until rate limit resets.

        Args:
            client_id: Client identifier.

        Returns:
            Seconds until full capacity is restored.
        """
        bucket = self._buckets[client_id]
        tokens_needed = self.burst_size - bucket.available_tokens
        return tokens_needed / self.refill_rate if tokens_needed > 0 else 0

    def block_client(self, client_id: str, duration: float) -> None:
        """
        Temporarily block a client.

        Args:
            client_id: Client identifier.
            duration: Block duration in seconds.
        """
        self._blocked_until[client_id] = time.time() + duration
        logger.warning(f"Client {client_id} blocked for {duration} seconds")

    def unblock_client(self, client_id: str) -> None:
        """
        Unblock a client.

        Args:
            client_id: Client identifier.
        """
        if client_id in self._blocked_until:
            del self._blocked_until[client_id]
            logger.info(f"Client {client_id} unblocked")

    def reset_client(self, client_id: str) -> None:
        """
        Reset rate limit for a client.

        Args:
            client_id: Client identifier.
        """
        if client_id in self._buckets:
            del self._buckets[client_id]
        self.unblock_client(client_id)

    def get_stats(self) -> dict:
        """
        Get rate limiter statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "enabled": self.enabled,
            "requests_per_minute": self.requests_per_minute,
            "burst_size": self.burst_size,
            "active_clients": len(self._buckets),
            "blocked_clients": len(self._blocked_until),
        }


# Global rate limiter instance
_rate_limiter: RateLimiter | None = None


def get_rate_limiter(
    requests_per_minute: int = 100,
    burst_size: int | None = None,
    enabled: bool = True,
) -> RateLimiter:
    """
    Get or create the global rate limiter instance.

    Args:
        requests_per_minute: Maximum requests per minute.
        burst_size: Maximum burst size.
        enabled: Whether rate limiting is enabled.

    Returns:
        RateLimiter instance.
    """
    global _rate_limiter

    if _rate_limiter is None:
        _rate_limiter = RateLimiter(
            requests_per_minute=requests_per_minute,
            burst_size=burst_size,
            enabled=enabled,
        )

    return _rate_limiter

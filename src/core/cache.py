"""
Caching Module for improved performance.
Supports in-memory and Redis caching.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

from loguru import logger


class CacheBackend(ABC):
    """Abstract base class for cache backends."""

    @abstractmethod
    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        pass

    @abstractmethod
    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        pass

    @abstractmethod
    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        pass

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Check if key exists in cache."""
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all cache entries."""
        pass


class InMemoryCache(CacheBackend):
    """
    Simple in-memory cache with TTL support.
    Suitable for single-instance deployments.
    """

    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        """
        Initialize in-memory cache.

        Args:
            max_size: Maximum number of entries.
            default_ttl: Default TTL in seconds.
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: dict[str, tuple[Any, float]] = {}
        self._access_times: dict[str, float] = {}

    def get(self, key: str) -> Any | None:
        """Get value from cache if not expired."""
        if key not in self._cache:
            return None

        value, expiry = self._cache[key]

        if expiry and time.time() > expiry:
            self.delete(key)
            return None

        self._access_times[key] = time.time()
        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache with optional TTL."""
        # Evict old entries if at capacity
        if len(self._cache) >= self.max_size and key not in self._cache:
            self._evict_lru()

        ttl = ttl or self.default_ttl
        expiry = time.time() + ttl if ttl else None

        self._cache[key] = (value, expiry)
        self._access_times[key] = time.time()

        return True

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        if key in self._cache:
            del self._cache[key]
            self._access_times.pop(key, None)
            return True
        return False

    def exists(self, key: str) -> bool:
        """Check if key exists and is not expired."""
        return self.get(key) is not None

    def clear(self) -> None:
        """Clear all cache entries."""
        self._cache.clear()
        self._access_times.clear()

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._access_times:
            return

        lru_key = min(self._access_times, key=self._access_times.get)
        self.delete(lru_key)

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
        }


class RedisCache(CacheBackend):
    """
    Redis-based cache for distributed deployments.
    Requires redis-py: pip install redis
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379/0",
        prefix: str = "rag:",
        default_ttl: int = 3600,
    ):
        """
        Initialize Redis cache.

        Args:
            url: Redis connection URL.
            prefix: Key prefix for namespacing.
            default_ttl: Default TTL in seconds.
        """
        self.url = url
        self.prefix = prefix
        self.default_ttl = default_ttl
        self._client = None

        self._connect()

    def _connect(self) -> None:
        """Connect to Redis."""
        try:
            import redis

            self._client = redis.from_url(self.url)
            self._client.ping()
            logger.info(f"Connected to Redis at {self.url}")
        except ImportError:
            logger.warning("redis-py not installed. Install with: pip install redis")
            self._client = None
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self._client = None

    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.prefix}{key}"

    def get(self, key: str) -> Any | None:
        """Get value from Redis."""
        if not self._client:
            return None

        try:
            value = self._client.get(self._make_key(key))
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in Redis with optional TTL."""
        if not self._client:
            return False

        try:
            ttl = ttl or self.default_ttl
            serialized = json.dumps(value)
            self._client.setex(self._make_key(key), ttl, serialized)
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete value from Redis."""
        if not self._client:
            return False

        try:
            return bool(self._client.delete(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False

    def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        if not self._client:
            return False

        try:
            return bool(self._client.exists(self._make_key(key)))
        except Exception as e:
            logger.error(f"Redis exists error: {e}")
            return False

    def clear(self) -> None:
        """Clear all cache entries with prefix."""
        if not self._client:
            return

        try:
            keys = self._client.keys(f"{self.prefix}*")
            if keys:
                self._client.delete(*keys)
        except Exception as e:
            logger.error(f"Redis clear error: {e}")

    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        if not self._client:
            return False
        try:
            self._client.ping()
            return True
        except Exception:
            return False


class CacheService:
    """
    High-level cache service with support for multiple backends.
    """

    def __init__(
        self,
        backend: CacheBackend | None = None,
        enabled: bool = True,
    ):
        """
        Initialize cache service.

        Args:
            backend: Cache backend to use.
            enabled: Whether caching is enabled.
        """
        self.enabled = enabled
        self.backend = backend or InMemoryCache()

        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Any | None:
        """Get value from cache."""
        if not self.enabled:
            return None

        value = self.backend.get(key)

        if value is not None:
            self._hits += 1
        else:
            self._misses += 1

        return value

    def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """Set value in cache."""
        if not self.enabled:
            return False

        return self.backend.set(key, value, ttl)

    def delete(self, key: str) -> bool:
        """Delete value from cache."""
        return self.backend.delete(key)

    def clear(self) -> None:
        """Clear cache."""
        self.backend.clear()
        self._hits = 0
        self._misses = 0

    def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: int | None = None,
    ) -> Any:
        """
        Get value from cache or compute and cache it.

        Args:
            key: Cache key.
            factory: Function to compute value if not cached.
            ttl: TTL for cached value.

        Returns:
            Cached or computed value.
        """
        value = self.get(key)
        if value is not None:
            return value

        value = factory()
        self.set(key, value, ttl)
        return value

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.2%}",
        }


def make_cache_key(*args, **kwargs) -> str:
    """
    Create a cache key from arguments.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        SHA256 hash of arguments.
    """
    key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()


# Global cache service instance
_cache_service: CacheService | None = None


def get_cache_service(
    backend_type: str = "memory",
    redis_url: str = "redis://localhost:6379/0",
    enabled: bool = True,
) -> CacheService:
    """
    Get or create the global cache service.

    Args:
        backend_type: Type of backend ("memory" or "redis").
        redis_url: Redis URL if using redis backend.
        enabled: Whether caching is enabled.

    Returns:
        CacheService instance.
    """
    global _cache_service

    if _cache_service is None:
        if backend_type == "redis":
            backend = RedisCache(url=redis_url)
            # Fall back to memory if Redis not available
            if not backend.is_connected():
                logger.warning("Redis not available, falling back to in-memory cache")
                backend = InMemoryCache()
        else:
            backend = InMemoryCache()

        _cache_service = CacheService(backend=backend, enabled=enabled)

    return _cache_service

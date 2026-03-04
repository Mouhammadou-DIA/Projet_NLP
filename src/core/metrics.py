"""
Metrics and Monitoring Module.
Provides Prometheus-compatible metrics for observability.
"""

import time
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Counter:
    """Simple counter metric."""

    name: str
    description: str
    value: int = 0
    labels: dict = field(default_factory=dict)

    def inc(self, amount: int = 1) -> None:
        """Increment counter."""
        self.value += amount

    def get(self) -> int:
        """Get current value."""
        return self.value


@dataclass
class Gauge:
    """Gauge metric that can go up and down."""

    name: str
    description: str
    value: float = 0.0
    labels: dict = field(default_factory=dict)

    def set(self, value: float) -> None:
        """Set gauge value."""
        self.value = value

    def inc(self, amount: float = 1.0) -> None:
        """Increment gauge."""
        self.value += amount

    def dec(self, amount: float = 1.0) -> None:
        """Decrement gauge."""
        self.value -= amount

    def get(self) -> float:
        """Get current value."""
        return self.value


@dataclass
class Histogram:
    """Histogram metric for measuring distributions."""

    name: str
    description: str
    buckets: tuple = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
    _values: list = field(default_factory=list)
    _sum: float = 0.0
    _count: int = 0

    def observe(self, value: float) -> None:
        """Record an observation."""
        self._values.append(value)
        self._sum += value
        self._count += 1

    def get_buckets(self) -> dict:
        """Get bucket counts."""
        bucket_counts = {}
        for bucket in self.buckets:
            bucket_counts[bucket] = sum(1 for v in self._values if v <= bucket)
        bucket_counts[float("inf")] = self._count
        return bucket_counts

    @property
    def sum(self) -> float:
        """Get sum of all observations."""
        return self._sum

    @property
    def count(self) -> int:
        """Get count of observations."""
        return self._count

    @property
    def mean(self) -> float:
        """Get mean of observations."""
        return self._sum / self._count if self._count > 0 else 0.0


class Timer:
    """Context manager for timing operations."""

    def __init__(self, histogram: Histogram | None = None):
        self.histogram = histogram
        self.start_time: float = 0
        self.elapsed: float = 0

    def __enter__(self) -> "Timer":
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self.start_time
        if self.histogram:
            self.histogram.observe(self.elapsed)


class MetricsRegistry:
    """
    Registry for all application metrics.
    Provides a central place to manage and export metrics.
    """

    def __init__(self):
        self._counters: dict[str, Counter] = {}
        self._gauges: dict[str, Gauge] = {}
        self._histograms: dict[str, Histogram] = {}

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self) -> None:
        """Initialize default application metrics."""
        # Request metrics
        self.register_counter("http_requests_total", "Total HTTP requests")
        self.register_counter("http_requests_errors_total", "Total HTTP request errors")
        self.register_histogram("http_request_duration_seconds", "HTTP request duration in seconds")

        # Chat metrics
        self.register_counter("chat_requests_total", "Total chat requests")
        self.register_counter("chat_requests_llm_total", "Total chat requests using LLM")
        self.register_histogram("chat_request_duration_seconds", "Chat request duration in seconds")

        # Embedding metrics
        self.register_counter("embeddings_generated_total", "Total embeddings generated")
        self.register_histogram("embedding_duration_seconds", "Embedding generation duration")

        # Search metrics
        self.register_counter("vector_searches_total", "Total vector searches")
        self.register_histogram("vector_search_duration_seconds", "Vector search duration")

        # LLM metrics
        self.register_counter("llm_requests_total", "Total LLM requests")
        self.register_counter("llm_errors_total", "Total LLM errors")
        self.register_histogram("llm_request_duration_seconds", "LLM request duration")

        # System metrics
        self.register_gauge("vector_store_documents", "Number of documents in vector store")
        self.register_gauge("active_connections", "Number of active connections")

    def register_counter(self, name: str, description: str) -> Counter:
        """Register a new counter metric."""
        if name not in self._counters:
            self._counters[name] = Counter(name=name, description=description)
        return self._counters[name]

    def register_gauge(self, name: str, description: str) -> Gauge:
        """Register a new gauge metric."""
        if name not in self._gauges:
            self._gauges[name] = Gauge(name=name, description=description)
        return self._gauges[name]

    def register_histogram(
        self, name: str, description: str, buckets: tuple | None = None
    ) -> Histogram:
        """Register a new histogram metric."""
        if name not in self._histograms:
            if buckets:
                self._histograms[name] = Histogram(
                    name=name, description=description, buckets=buckets
                )
            else:
                self._histograms[name] = Histogram(name=name, description=description)
        return self._histograms[name]

    def counter(self, name: str) -> Counter:
        """Get a counter by name."""
        return self._counters.get(name, Counter(name=name, description=""))

    def gauge(self, name: str) -> Gauge:
        """Get a gauge by name."""
        return self._gauges.get(name, Gauge(name=name, description=""))

    def histogram(self, name: str) -> Histogram:
        """Get a histogram by name."""
        return self._histograms.get(name, Histogram(name=name, description=""))

    def timer(self, histogram_name: str) -> Timer:
        """Create a timer for a histogram."""
        histogram = self._histograms.get(histogram_name)
        return Timer(histogram)

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Returns:
            Prometheus-compatible metrics string.
        """
        lines = []

        # Export counters
        for name, counter in self._counters.items():
            lines.append(f"# HELP {name} {counter.description}")
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {counter.value}")

        # Export gauges
        for name, gauge in self._gauges.items():
            lines.append(f"# HELP {name} {gauge.description}")
            lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {gauge.value}")

        # Export histograms
        for name, histogram in self._histograms.items():
            lines.append(f"# HELP {name} {histogram.description}")
            lines.append(f"# TYPE {name} histogram")

            buckets = histogram.get_buckets()
            for bucket, count in buckets.items():
                if bucket == float("inf"):
                    lines.append(f'{name}_bucket{{le="+Inf"}} {count}')
                else:
                    lines.append(f'{name}_bucket{{le="{bucket}"}} {count}')

            lines.append(f"{name}_sum {histogram.sum}")
            lines.append(f"{name}_count {histogram.count}")

        return "\n".join(lines)

    def export_json(self) -> dict:
        """
        Export metrics as JSON.

        Returns:
            Dictionary with all metrics.
        """
        return {
            "counters": {
                name: {"value": c.value, "description": c.description}
                for name, c in self._counters.items()
            },
            "gauges": {
                name: {"value": g.value, "description": g.description}
                for name, g in self._gauges.items()
            },
            "histograms": {
                name: {
                    "count": h.count,
                    "sum": h.sum,
                    "mean": h.mean,
                    "description": h.description,
                }
                for name, h in self._histograms.items()
            },
        }

    def reset(self) -> None:
        """Reset all metrics."""
        for counter in self._counters.values():
            counter.value = 0
        for gauge in self._gauges.values():
            gauge.value = 0
        for histogram in self._histograms.values():
            histogram._values = []
            histogram._sum = 0
            histogram._count = 0


# Global metrics registry
_metrics: MetricsRegistry | None = None


def get_metrics() -> MetricsRegistry:
    """Get or create the global metrics registry."""
    global _metrics
    if _metrics is None:
        _metrics = MetricsRegistry()
    return _metrics


def timed(histogram_name: str) -> Callable:
    """
    Decorator to time function execution.

    Args:
        histogram_name: Name of the histogram to record to.

    Returns:
        Decorated function.
    """

    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            metrics = get_metrics()
            with metrics.timer(histogram_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator

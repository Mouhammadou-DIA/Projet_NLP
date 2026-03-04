"""
Benchmark Script - Reddit RAG Chatbot
Performance testing and benchmarking
"""

import statistics
import sys
import time
from pathlib import Path


# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.logging_config import get_logger, log_startup
from src.models.schemas import ChatRequest
from src.services.chatbot_service import get_chatbot_service


logger = get_logger(__name__)


class Benchmark:
    """Benchmark suite for chatbot performance"""

    def __init__(self):
        """Initialize benchmark"""
        self.chatbot = get_chatbot_service()
        self.results = []

    def run_single_query(self, query: str, use_llm: bool = False) -> dict:
        """
        Run single query benchmark

        Args:
            query: Query text
            use_llm: Whether to use LLM

        Returns:
            Benchmark results
        """
        request = ChatRequest(message=query, use_llm=use_llm, n_results=5)

        start_time = time.time()
        response = self.chatbot.chat(request)
        duration = (time.time() - start_time) * 1000  # ms

        return {
            "query": query,
            "duration_ms": duration,
            "use_llm": use_llm,
            "response_length": len(response.message),
            "n_sources": len(response.sources),
        }

    def run_batch(self, queries: list[str], use_llm: bool = False, iterations: int = 3):
        """
        Run batch benchmark

        Args:
            queries: List of queries
            use_llm: Whether to use LLM
            iterations: Number of iterations per query
        """
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Benchmark: {'LLM Mode' if use_llm else 'Simple Mode'}")
        logger.info(f"Queries: {len(queries)}, Iterations: {iterations}")
        logger.info(f"{'=' * 60}\n")

        all_durations = []

        for query in queries:
            logger.info(f"Query: '{query[:50]}...'")
            query_durations = []

            for i in range(iterations):
                result = self.run_single_query(query, use_llm)
                query_durations.append(result["duration_ms"])
                logger.info(f"  Iteration {i + 1}: {result['duration_ms']:.2f}ms")

            avg_duration = statistics.mean(query_durations)
            logger.info(f"  Average: {avg_duration:.2f}ms\n")

            all_durations.extend(query_durations)
            self.results.append(
                {
                    "query": query,
                    "iterations": iterations,
                    "durations": query_durations,
                    "avg_duration": avg_duration,
                    "min_duration": min(query_durations),
                    "max_duration": max(query_durations),
                }
            )

        return all_durations

    def print_summary(self, durations: list[float], mode: str):
        """Print benchmark summary"""
        logger.info(f"\n{'=' * 60}")
        logger.info(f"SUMMARY - {mode}")
        logger.info(f"{'=' * 60}")
        logger.info(f"Total queries: {len(durations)}")
        logger.info(f"Mean: {statistics.mean(durations):.2f}ms")
        logger.info(f"Median: {statistics.median(durations):.2f}ms")
        logger.info(f"Min: {min(durations):.2f}ms")
        logger.info(f"Max: {max(durations):.2f}ms")
        logger.info(f"Std Dev: {statistics.stdev(durations):.2f}ms")

        # Percentiles
        sorted_durations = sorted(durations)
        p50 = sorted_durations[len(sorted_durations) // 2]
        p95 = sorted_durations[int(len(sorted_durations) * 0.95)]
        p99 = sorted_durations[int(len(sorted_durations) * 0.99)]

        logger.info(f"P50: {p50:.2f}ms")
        logger.info(f"P95: {p95:.2f}ms")
        logger.info(f"P99: {p99:.2f}ms")
        logger.info(f"{'=' * 60}\n")


def main():
    """Main benchmark function"""
    log_startup()

    logger.info("Starting Performance Benchmark")

    # Test queries
    test_queries = [
        "What phone should I buy?",
        "I'm feeling sad today",
        "How do I make friends?",
        "I just got a new job",
        "Quel téléphone acheter?",
    ]

    benchmark = Benchmark()

    # Benchmark simple mode
    logger.info("\nBenchmarking SIMPLE mode...")
    simple_durations = benchmark.run_batch(test_queries, use_llm=False, iterations=3)
    benchmark.print_summary(simple_durations, "Simple Mode")

    # Benchmark LLM mode (if available)
    if benchmark.chatbot.llm_service.is_available():
        logger.info("\nBenchmarking LLM mode...")
        response = input("This will be slower. Continue? (yes/no): ")
        if response.lower() == "yes":
            llm_durations = benchmark.run_batch(test_queries, use_llm=True, iterations=2)
            benchmark.print_summary(llm_durations, "LLM Mode")
    else:
        logger.warning("LLM not available, skipping LLM benchmark")

    logger.info("\nBenchmark complete!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nBenchmark interrupted")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Benchmark failed: {e!s}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

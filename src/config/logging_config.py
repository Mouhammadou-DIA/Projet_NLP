"""
Logging Configuration - Professional Reddit RAG Chatbot
Structured logging with rotation, filtering, and monitoring integration
"""

import sys
from pathlib import Path

from loguru import logger

from src.config.settings import settings


class LogConfig:
    """
    Professional logging configuration

    Features:
    - Structured JSON logging for production
    - Human-readable logs for development
    - Log rotation and retention
    - Multiple output destinations
    - Integration with monitoring tools
    """

    def __init__(self):
        """Initialize logging configuration"""
        self.setup_logging()

    def setup_logging(self):
        """Configure loguru logger"""
        # Remove default handler
        logger.remove()

        # Console handler (development)
        if settings.ENVIRONMENT == "development":
            logger.add(
                sys.stdout,
                format=self._get_console_format(),
                level=settings.LOG_LEVEL,
                colorize=True,
                backtrace=True,
                diagnose=True,
            )

        # File handler (always)
        if settings.LOG_FILE:
            logger.add(
                settings.LOG_FILE,
                format=self._get_file_format(),
                level=settings.LOG_LEVEL,
                rotation=settings.LOG_ROTATION,
                retention=settings.LOG_RETENTION,
                compression="zip",
                serialize=settings.LOG_FORMAT == "json",
                backtrace=True,
                diagnose=True,
            )

        # Error file handler (errors only)
        error_log = Path(settings.LOG_FILE).parent / "error.log" if settings.LOG_FILE else None
        if error_log:
            logger.add(
                error_log,
                format=self._get_file_format(),
                level="ERROR",
                rotation=settings.LOG_ROTATION,
                retention=settings.LOG_RETENTION,
                compression="zip",
                serialize=settings.LOG_FORMAT == "json",
                backtrace=True,
                diagnose=True,
            )

        # Production console (JSON)
        if settings.ENVIRONMENT == "production":
            logger.add(
                sys.stdout,
                format=self._get_json_format(),
                level=settings.LOG_LEVEL,
                serialize=True,
            )

    @staticmethod
    def _get_console_format() -> str:
        """Get console log format (development)"""
        return (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        )

    @staticmethod
    def _get_file_format() -> str:
        """Get file log format"""
        return "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}"

    @staticmethod
    def _get_json_format() -> str:
        """Get JSON log format (production)"""
        return "{message}"

    @staticmethod
    def get_logger(name: str | None = None):
        """
        Get logger instance

        Args:
            name: Logger name (usually __name__)

        Returns:
            Logger instance
        """
        if name:
            return logger.bind(module=name)
        return logger


# Initialize logging
log_config = LogConfig()


# Export configured logger
def get_logger(name: str | None = None):
    """Get logger for module"""
    return log_config.get_logger(name)


# Convenience functions
def log_startup():
    """Log application startup"""
    logger.info("=" * 60)
    logger.info(f" Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"Debug: {settings.DEBUG}")
    logger.info(f"Log Level: {settings.LOG_LEVEL}")
    logger.info("=" * 60)


def log_shutdown():
    """Log application shutdown"""
    logger.info("=" * 60)
    logger.info(f" Shutting down {settings.APP_NAME}")
    logger.info("=" * 60)


def log_error(error: Exception, context: dict | None = None):
    """
    Log error with context

    Args:
        error: Exception to log
        context: Additional context dictionary
    """
    logger.error(f"Error occurred: {error!s}")
    if context:
        logger.error(f"Context: {context}")
    logger.exception(error)


def log_metric(metric_name: str, value: float, tags: dict | None = None):
    """
    Log metric (for monitoring)

    Args:
        metric_name: Name of the metric
        value: Metric value
        tags: Additional tags
    """
    log_data = {"metric": metric_name, "value": value, "tags": tags or {}}
    logger.info(f"METRIC: {log_data}")


def log_request(method: str, path: str, status_code: int, duration: float):
    """
    Log HTTP request

    Args:
        method: HTTP method
        path: Request path
        status_code: Response status code
        duration: Request duration in ms
    """
    logger.info(
        f"HTTP {method} {path} - {status_code} - {duration:.2f}ms",
        extra={
            "http_method": method,
            "http_path": path,
            "http_status": status_code,
            "duration_ms": duration,
        },
    )

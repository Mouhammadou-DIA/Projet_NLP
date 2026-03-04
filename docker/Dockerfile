# ============================================================
# REDDIT RAG CHATBOT - PRODUCTION DOCKERFILE
# Multi-stage build for optimized image size
# ============================================================

# ==================== BUILDER STAGE ====================
FROM python:3.12-slim as builder

# Set working directory
WORKDIR /build

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Create virtual environment and install dependencies
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ==================== RUNTIME STAGE ====================
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    ENVIRONMENT=production

# Create non-root user
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data /app/logs && \
    chown -R appuser:appuser /app

# Set working directory
WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser api/ api/
COPY --chown=appuser:appuser frontend/ frontend/
COPY --chown=appuser:appuser scripts/ scripts/
COPY --chown=appuser:appuser run_api.py .
COPY --chown=appuser:appuser run_frontend.py .
COPY --chown=appuser:appuser .env* .

# Switch to non-root user
USER appuser

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health/live')" || exit 1

# Default command (API server - also serves frontend static files)
CMD ["python", "run_api.py"]

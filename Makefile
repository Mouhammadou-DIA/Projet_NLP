# =============================================================================
# Makefile for Reddit RAG Chatbot
# =============================================================================

.PHONY: help install install-dev setup clean lint format test test-unit test-integration coverage run-api run-ui docker-build docker-up docker-down prepare-data index benchmark

# Default target
.DEFAULT_GOAL := help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
RUFF := ruff
MYPY := mypy
DOCKER_COMPOSE := docker-compose -f docker/docker-compose.yml

# Colors for output
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
NC := \033[0m # No Color

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "$(BLUE)Reddit RAG Chatbot - Available Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -r requirements.txt

install-dev: ## Install development dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

setup: install-dev ## Full development setup
	@echo "$(GREEN)Creating necessary directories...$(NC)"
	mkdir -p data/raw data/processed data/vector_db logs
	@echo "$(GREEN)Copying environment template...$(NC)"
	cp -n .env.example .env || true
	@echo "$(GREEN)Setup complete!$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run linting checks
	@echo "$(BLUE)Running Ruff linter...$(NC)"
	$(RUFF) check .
	@echo "$(BLUE)Running Ruff formatter check...$(NC)"
	$(RUFF) format --check .

format: ## Format code with Ruff
	@echo "$(BLUE)Formatting code...$(NC)"
	$(RUFF) format .
	$(RUFF) check --fix .

type-check: ## Run type checking with MyPy
	@echo "$(BLUE)Running MyPy...$(NC)"
	$(MYPY) src/ --ignore-missing-imports

check: lint type-check ## Run all code quality checks

# =============================================================================
# Testing
# =============================================================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(NC)"
	$(PYTEST) tests/ -v

test-unit: ## Run unit tests only
	@echo "$(BLUE)Running unit tests...$(NC)"
	$(PYTEST) tests/unit/ -v -m unit

test-integration: ## Run integration tests only
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(PYTEST) tests/integration/ -v -m integration

test-fast: ## Run tests excluding slow ones
	@echo "$(BLUE)Running fast tests...$(NC)"
	$(PYTEST) tests/ -v -m "not slow"

coverage: ## Run tests with coverage report
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) tests/ --cov=src --cov-report=html --cov-report=term-missing
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

# =============================================================================
# Application
# =============================================================================

run-api: ## Start the FastAPI server
	@echo "$(BLUE)Starting API server...$(NC)"
	$(PYTHON) run_api.py

run-ui: ## Start the Gradio UI
	@echo "$(BLUE)Starting Gradio UI...$(NC)"
	$(PYTHON) run_ui.py

run-streamlit: ## Start the Streamlit UI
	@echo "$(BLUE)Starting Streamlit UI...$(NC)"
	streamlit run ui/streamlit_app.py

run: ## Start both API and UI (requires two terminals)
	@echo "$(YELLOW)Run 'make run-api' in one terminal and 'make run-ui' in another$(NC)"

# =============================================================================
# Data Processing
# =============================================================================

prepare-data: ## Prepare and clean raw data
	@echo "$(BLUE)Preparing data...$(NC)"
	$(PYTHON) scripts/prepare_data.py

index: ## Index conversations into vector store
	@echo "$(BLUE)Indexing conversations...$(NC)"
	$(PYTHON) scripts/index_conversations.py

benchmark: ## Run performance benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(PYTHON) scripts/benchmark.py

data-pipeline: prepare-data index ## Run full data pipeline
	@echo "$(GREEN)Data pipeline complete!$(NC)"

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(NC)"
	$(DOCKER_COMPOSE) build

docker-up: ## Start Docker containers
	@echo "$(BLUE)Starting Docker containers...$(NC)"
	$(DOCKER_COMPOSE) up -d

docker-down: ## Stop Docker containers
	@echo "$(BLUE)Stopping Docker containers...$(NC)"
	$(DOCKER_COMPOSE) down

docker-logs: ## View Docker logs
	$(DOCKER_COMPOSE) logs -f

docker-restart: docker-down docker-up ## Restart Docker containers

docker-clean: ## Remove Docker containers and volumes
	@echo "$(RED)Removing Docker containers and volumes...$(NC)"
	$(DOCKER_COMPOSE) down -v --remove-orphans

# =============================================================================
# Cleanup
# =============================================================================

clean: ## Clean build artifacts and cache
	@echo "$(BLUE)Cleaning up...$(NC)"
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache
	rm -rf htmlcov .coverage coverage.xml
	rm -rf *.egg-info build dist
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "$(GREEN)Cleanup complete!$(NC)"

clean-data: ## Clean processed data (keeps raw data)
	@echo "$(RED)Cleaning processed data...$(NC)"
	rm -rf data/processed/* data/vector_db/*
	@echo "$(GREEN)Processed data cleaned!$(NC)"

clean-all: clean clean-data ## Clean everything including data
	@echo "$(GREEN)Full cleanup complete!$(NC)"

# =============================================================================
# Development Utilities
# =============================================================================

shell: ## Start Python shell with project context
	@echo "$(BLUE)Starting Python shell...$(NC)"
	$(PYTHON) -i -c "from src.services.chatbot_service import ChatbotService; print('ChatbotService available')"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking dependencies...$(NC)"
	$(PIP) list --outdated

update-deps: ## Update dependencies
	@echo "$(BLUE)Updating dependencies...$(NC)"
	$(PIP) install --upgrade -r requirements.txt

security-check: ## Run security checks
	@echo "$(BLUE)Running security checks...$(NC)"
	$(PIP) install bandit safety
	bandit -r src/ -ll
	safety check -r requirements.txt || true

# =============================================================================
# Documentation
# =============================================================================

docs-serve: ## Serve documentation locally
	@echo "$(BLUE)Serving documentation...$(NC)"
	$(PYTHON) -m http.server 8080 --directory docs/

# =============================================================================
# CI/CD Helpers
# =============================================================================

ci: lint type-check test ## Run CI pipeline locally
	@echo "$(GREEN)CI pipeline complete!$(NC)"

pre-commit: ## Run pre-commit hooks
	pre-commit run --all-files

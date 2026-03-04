# Contributing Guide

Thank you for your interest in contributing to the Reddit RAG Chatbot project!

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Review Process](#review-process)

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help others learn and grow

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch
5. Make your changes
6. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.10+
- Git
- (Optional) Docker

### Installation

```bash
# Clone your fork
git clone https://github.com/ManoDek02/Projet-NLP
cd reddit-rag-chatbot

# Add upstream remote
git remote add upstream https://github.com/ManoDek02/Projet-NLP

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Copy environment file
cp .env.example .env
```

### Verify Setup

```bash
# Run tests
pytest

# Run linting
ruff check .

# Run type checking
mypy src/
```

## Making Changes

### Branch Naming

Use descriptive branch names:

- `feature/add-streaming-support`
- `fix/search-score-calculation`
- `docs/update-api-documentation`
- `refactor/embeddings-service`

### Commit Messages

Follow conventional commits:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

Examples:
```
feat(api): add streaming response support

fix(embeddings): handle empty text input gracefully

docs(readme): update installation instructions
```

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints
- Maximum line length: 100 characters
- Use docstrings for public functions

```python
def search_similar(
    self,
    query: str,
    n_results: int = 5,
    min_score: float = 0.5,
) -> list[SearchResult]:
    """Search for similar conversations.

    Args:
        query: The search query text.
        n_results: Maximum number of results to return.
        min_score: Minimum similarity score threshold.

    Returns:
        List of SearchResult objects sorted by relevance.

    Raises:
        ValueError: If query is empty or n_results < 1.
    """
    ...
```

### Project Structure

```
src/
├── config/      # Configuration
├── core/        # Core components (embeddings, vector store, LLM)
├── services/    # Business logic
├── models/      # Pydantic schemas
└── utils/       # Utilities
```

### Error Handling

```python
# Use specific exceptions
class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass

# Provide context
try:
    embedding = self.model.encode(text)
except Exception as e:
    raise EmbeddingError(f"Failed to embed text: {e}") from e
```

### Logging

```python
from loguru import logger

# Use appropriate log levels
logger.debug("Processing batch of {n} texts", n=len(texts))
logger.info("Successfully indexed {count} conversations", count=count)
logger.warning("LLM timeout, falling back to simple response")
logger.error("Vector store connection failed: {error}", error=str(e))
```

## Testing

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data
└── conftest.py     # Shared fixtures
```

### Writing Tests

```python
import pytest
from src.core.embeddings import EmbeddingService

class TestEmbeddingService:
    @pytest.fixture
    def service(self):
        return EmbeddingService()

    def test_embed_text_returns_correct_dimension(self, service):
        embedding = service.embed_text("Hello world")
        assert len(embedding) == 384

    def test_embed_empty_text_raises_error(self, service):
        with pytest.raises(ValueError):
            service.embed_text("")

    @pytest.mark.slow
    def test_embed_batch_large_dataset(self, service):
        texts = ["text"] * 1000
        embeddings = service.embed_batch(texts)
        assert len(embeddings) == 1000
```

### Running Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=src --cov-report=html

# Specific test file
pytest tests/unit/test_embeddings.py

# Skip slow tests
pytest -m "not slow"

# Verbose output
pytest -v
```

### Test Coverage

Maintain minimum 70% coverage. Check with:

```bash
pytest --cov=src --cov-fail-under=70
```

## Submitting Changes

### Before Submitting

1. Ensure all tests pass
2. Run linting and fix issues
3. Update documentation if needed
4. Add tests for new features

```bash
# Pre-submission checklist
pytest
ruff check .
ruff format .
mypy src/
```

### Pull Request

1. Push your branch to your fork
2. Open a PR against `main`
3. Fill out the PR template
4. Request review

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests added/updated
- [ ] All tests passing

## Checklist
- [ ] Code follows project style
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

## Review Process

1. **Automated Checks**: CI runs tests and linting
2. **Code Review**: Maintainer reviews changes
3. **Feedback**: Address any requested changes
4. **Approval**: Once approved, PR is merged

### Review Criteria

- Code quality and readability
- Test coverage
- Documentation
- Performance implications
- Security considerations

## Questions?

- Open an issue for bugs or feature requests
- Discussions for general questions
- Tag maintainers for urgent issues

Thank you for contributing!

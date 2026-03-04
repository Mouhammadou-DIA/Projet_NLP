# API Documentation

## Base URL

```
http://localhost:8000
```

## Authentication

Authentication is optional and disabled by default. Enable with `ENABLE_AUTH=true`.

When enabled, include the JWT token in the Authorization header:
```
Authorization: Bearer <token>
```

## Endpoints

### Root

#### GET /

Returns application information.

**Response**
```json
{
  "name": "Reddit RAG Chatbot API",
  "version": "1.0.0",
  "status": "running",
  "docs_url": "/docs"
}
```

---

### Chat

#### POST /api/v1/chat/

Send a message to the chatbot.

**Request Body**
```json
{
  "message": "What do you think about artificial intelligence?",
  "use_llm": false,
  "n_results": 5,
  "temperature": 0.7,
  "max_tokens": 500
}
```

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `message` | string | Yes | - | User's message (1-1000 chars) |
| `use_llm` | boolean | No | false | Use LLM for response generation |
| `n_results` | integer | No | 5 | Number of context results (1-20) |
| `temperature` | float | No | 0.7 | LLM temperature (0.0-2.0) |
| `max_tokens` | integer | No | 500 | Max response tokens (1-2000) |

**Response**
```json
{
  "message": "I think AI is fascinating...",
  "sources": [
    {
      "context": "Original question about AI...",
      "response": "Original response...",
      "score": 0.85,
      "rank": 1
    }
  ],
  "metadata": {
    "duration_ms": 125,
    "mode": "simple",
    "n_sources": 5,
    "model": null
  }
}
```

**Error Responses**

| Status | Description |
|--------|-------------|
| 400 | Invalid request (message too long, invalid parameters) |
| 500 | Internal server error |

---

#### GET /api/v1/chat/stats

Get chatbot statistics.

**Response**
```json
{
  "total_conversations": 56295,
  "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
  "embedding_dimension": 384,
  "vector_store_type": "chromadb",
  "llm_provider": "ollama",
  "llm_model": "llama3.2"
}
```

---

#### GET /api/v1/chat/examples

Get example questions.

**Response**
```json
{
  "examples": {
    "french": [
      "Comment vas-tu aujourd'hui ?",
      "Qu'est-ce que tu penses de la technologie ?",
      "Raconte-moi une blague"
    ],
    "english": [
      "How are you doing today?",
      "What do you think about technology?",
      "Tell me a joke"
    ]
  }
}
```

---

### Health

#### GET /api/v1/health/

Full health check of all components.

**Response**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T10:30:00Z",
  "components": {
    "embedding_service": {
      "status": "healthy",
      "model": "paraphrase-multilingual-MiniLM-L12-v2"
    },
    "vector_store": {
      "status": "healthy",
      "type": "chromadb",
      "document_count": 56295
    },
    "llm_service": {
      "status": "healthy",
      "provider": "ollama",
      "model": "llama3.2"
    }
  }
}
```

---

#### GET /api/v1/health/ready

Kubernetes readiness probe.

**Response (200 OK)**
```json
{
  "status": "ready"
}
```

**Response (503 Service Unavailable)**
```json
{
  "status": "not_ready",
  "reason": "Vector store not initialized"
}
```

---

#### GET /api/v1/health/live

Kubernetes liveness probe.

**Response (200 OK)**
```json
{
  "status": "alive"
}
```

---

#### GET /api/v1/health/version

Get version information.

**Response**
```json
{
  "version": "1.0.0",
  "python_version": "3.10.12",
  "environment": "development"
}
```

---

## Error Handling

All errors follow a consistent format:

```json
{
  "detail": "Error message describing the issue"
}
```

### Common Error Codes

| Status Code | Description |
|-------------|-------------|
| 400 | Bad Request - Invalid input |
| 401 | Unauthorized - Missing or invalid token |
| 404 | Not Found - Endpoint doesn't exist |
| 422 | Unprocessable Entity - Validation error |
| 429 | Too Many Requests - Rate limit exceeded |
| 500 | Internal Server Error |
| 503 | Service Unavailable |

---

## Rate Limiting

When rate limiting is enabled (`RATE_LIMIT_ENABLED=true`):

- **Default limit**: 100 requests per 60 seconds
- **Headers included in response**:
  - `X-RateLimit-Limit`: Maximum requests allowed
  - `X-RateLimit-Remaining`: Remaining requests
  - `X-RateLimit-Reset`: Time until limit resets

---

## Examples

### cURL

```bash
# Simple chat
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, how are you?"}'

# Chat with LLM
curl -X POST http://localhost:8000/api/v1/chat/ \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is machine learning?",
    "use_llm": true,
    "n_results": 3
  }'

# Health check
curl http://localhost:8000/api/v1/health/

# Get stats
curl http://localhost:8000/api/v1/chat/stats
```

### Python

```python
import requests

# Simple chat
response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={"message": "Hello!"}
)
print(response.json())

# Chat with LLM
response = requests.post(
    "http://localhost:8000/api/v1/chat/",
    json={
        "message": "Explain quantum computing",
        "use_llm": True,
        "n_results": 5,
        "temperature": 0.8
    }
)
result = response.json()
print(f"Response: {result['message']}")
print(f"Sources: {len(result['sources'])}")
```

### JavaScript

```javascript
// Simple chat
const response = await fetch('http://localhost:8000/api/v1/chat/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ message: 'Hello!' })
});
const data = await response.json();
console.log(data.message);

// Chat with LLM
const llmResponse = await fetch('http://localhost:8000/api/v1/chat/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    message: 'What is the meaning of life?',
    use_llm: true,
    n_results: 3
  })
});
const llmData = await llmResponse.json();
console.log(llmData);
```

---

## OpenAPI Documentation

Interactive API documentation is available at:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

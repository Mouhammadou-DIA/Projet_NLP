# Reddit RAG Chatbot

RAG (Retrieval-Augmented Generation) chatbot built on Reddit conversations, exposed through a FastAPI backend and a lightweight HTML/CSS/JS frontend.

This project demonstrates a modular, production-ready RAG architecture with multi-LLM support and Docker deployment.

---

## Features

- Full RAG pipeline (Embedding → Vector Search → Context Injection → LLM)
- Multilingual embeddings (`paraphrase-multilingual-MiniLM-L12-v2`)
- Multiple LLM providers:
  - Ollama (local)
  - OpenAI
  - Anthropic
  - Groq
- Reranking support
- Response caching
- Built-in rate limiting
- REST API (FastAPI)
- Static frontend interface
- Docker deployment ready
- Unit and integration tests

---

## Architecture

### Offline Indexing
Reddit CSV
↓
Cleaning & chunking
↓
Embeddings (SentenceTransformers)
↓
ChromaDB (Vector Store)


### Online Inference


User question
↓
Embedding
↓
Similarity search (Top-K)
↓
Context injection
↓
LLM generation
↓
Response


Layered architecture:


API (FastAPI)
↓
Services
↓
Core (RAG components)
↓
Vector DB + LLM


---

## Project Structure


Projet_NLP/
│
├── api/
│ ├── main.py
│ ├── routes/
│ └── schemas/
│
├── src/
│ ├── config/
│ ├── core/
│ ├── services/
│ ├── models/
│ └── utils/
│
├── frontend/
├── scripts/
├── docker/
├── data/
├── tests/
└── docs/


---

## Installation

### Clone the repository


git clone https://github.com/Mouhammadou-DIA/Projet_NLP.git
cd Projet_NLP
Create virtual environment

Linux / Mac:

python -m venv venv
source venv/bin/activate

Windows:

python -m venv venv
venv\Scripts\activate
Install dependencies
pip install -r requirements.txt
Configure environment variables
cp .env.example .env

Example .env:

API_HOST=0.0.0.0
API_PORT=8000

LLM_PROVIDER=ollama
LLM_MODEL=llama3.1:8b

EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHROMA_PERSIST_DIRECTORY=./data/vector_db
Data Preparation
python scripts/prepare_data.py
python scripts/index_conversations.py
Run the Application

Start the API:

python run_api.py

API documentation:

http://localhost:8000/docs

Start the frontend (in a new terminal):

python run_frontend.py

Frontend:

http://localhost:3000
API Usage
Chat endpoint
curl -X POST http://localhost:8000/api/v1/chat/ \
-H "Content-Type: application/json" \
-d '{
  "message": "What do you think about AI?",
  "use_llm": true,
  "n_results": 5,
  "temperature": 0.7,
  "max_tokens": 500
}'
Health check
curl http://localhost:8000/api/v1/health/
Docker Deployment
docker-compose -f docker/docker-compose.yml up -d

View logs:

docker-compose -f docker/docker-compose.yml logs -f
Development

Install development dependencies:

pip install -e ".[dev]"

Run tests:

pytest

Lint:

ruff check .
ruff format .

Type checking:

mypy src/
Benchmark
python scripts/benchmark.py
Contributing

Fork the repository

Create a feature branch

Commit your changes

Push your branch

Open a Pull Request

Author

Mouhammadou DIA
